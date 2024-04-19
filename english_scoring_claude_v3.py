import base64
from mimetypes import guess_type
import os, io, time
import uuid
from PIL import Image
import streamlit as st

import json
import boto3
import logging
from botocore.exceptions import ClientError
# from dotenv import load_dotenv
# load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REGION = os.getenv("REGION")
AWS_ACCESS_KEY_ID=os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY=os.getenv("AWS_SECRET_ACCESS_KEY")
BEDROCK_CLAUDE_MODEL_NAME=os.getenv("BEDROCK_CLAUDE_MODEL_NAME")

MAX_RESPONSE_TOKEN = int(os.getenv("MAX_RESPONSE_TOKEN"))


class BedrockRuntimeWrapper:
  """Encapsulates Amazon Bedrock Runtime actions."""

  def __init__(self, client=None):
      """
      :param bedrock_runtime_client: A low-level client representing Amazon Bedrock Runtime.
                                      Describes the API operations for running inference using
                                      Bedrock models.
      """
      self.bedrock_runtime_client = client or boto3.client(service_name="bedrock-runtime", region_name=REGION)

  def invoke_claude(self, prompt):
      """
      Invokes the Anthropic Claude 2 model to run an inference using the input
      provided in the request body.

      :param prompt: The prompt that you want Claude to complete.
      :return: Inference response from the model.
      """

      try:
          # The different model providers have individual request and response formats.
          # For the format, ranges, and default values for Anthropic Claude, refer to:
          # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html

          body = {
              "prompt": prompt,
              "max_tokens_to_sample": MAX_RESPONSE_TOKEN,
              "temperature": 0.9,
          }

          response = self.bedrock_runtime_client.invoke_model(
              modelId=BEDROCK_CLAUDE_MODEL_NAME, body=json.dumps(body)
          )

          response_body = json.loads(response["body"].read())
          completion = response_body["completion"]

          return completion

      except ClientError:
          logger.error("Couldn't invoke Anthropic Claude")
          raise

  def invoke_model_with_response_stream(self, prompt):
      """
      Invokes the Anthropic Claude 2 model to run an inference and process the response stream.

      :param prompt: The prompt that you want Claude to complete.
      :return: Inference response from the model.
      """

      try:
          # The different model providers have individual request and response formats.
          # For the format, ranges, and default values for Anthropic Claude, refer to:
          # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html

          body = {
              "prompt": prompt,
              "max_tokens_to_sample": 1024,
              "temperature": 0.9,
          }

          response = self.bedrock_runtime_client.invoke_model_with_response_stream(
              modelId=BEDROCK_CLAUDE_MODEL_NAME, body=json.dumps(body)
          )

          return response
          # for event in response.get("body"):
          #     chunk = json.loads(event["chunk"]["bytes"])["completion"]
          #     yield chunk

      except ClientError:
          logger.error("Couldn't invoke Anthropic Claude v2")
          raise
      
  def invoke_claude_3_multimodal(self, system_content, user_content, image_source=None):
    """
    Invokes Anthropic Claude 3 Sonnet to run a multimodal inference using the input
    provided in the request body.

    :param prompt:            The prompt that you want Claude 3 to use.
    :param base64_image_data: The base64-encoded image that you want to add to the request.
    :return: Inference response from the model.
    """

    # Invoke the model with the prompt and the encoded image
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "system": system_content,
        "messages": [
          {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_content,
                },
            ],
          }
        ],
    }
    if image_source:
      request_body["messages"][0]["content"].append({
          "type": "image",
          "source": image_source,
      })

    try:
        response = self.bedrock_runtime_client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
        )

        # Process and print the response
        result = json.loads(response.get("body").read())
        input_tokens = result["usage"]["input_tokens"]
        output_tokens = result["usage"]["output_tokens"]
        # output_list = result.get("content", [])

        print("Invocation details:")
        print(f"- The input length is {input_tokens} tokens.")
        print(f"- The output length is {output_tokens} tokens.")

        # print(f"- The model returned {len(output_list)} response(s):")
        # for output in output_list:
        #     print(output["text"])

        return result
    except ClientError as err:
        logger.error(
            "Couldn't invoke Claude 3 Sonnet. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise
    
# Function to encode a local image into base64 data
def local_image_to_base64(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    return {
      "type": "base64",
      "media_type": f"{mime_type}",
      "data": f"{base64_encoded_data}",
    }

    # Construct the data URL
    # return f"data:{mime_type};base64,{base64_encoded_data}"


# Keys: Question, Answer
system_content_writing_task_1_assessment = '''
You are an experienced IELTS examiner tasked with evaluating a candidate's IELTS Writing Task 1 response, which may describe visual data (e.g., graphs, charts), processes (how something works or is done), or objects/events (e.g., a building layout). Your analysis must be grounded in the IELTS Writing Task 1 rubric standards, and feedback should be objective and focused on constructive criticism. Alongside identifying areas for improvement, also acknowledge any effective structures or strategies employed by the candidate.
Please provide a meticulous assessment of the following criteria: task achievement, coherence and cohesion, lexical resource, grammatical range and accuracy. Compare the candidate's response to the task requirements, commenting on the clarity and accuracy of the language used in relation to the provided visual data or description.
Highlight the range and accuracy of vocabulary, noting any errors that may impact comprehension, and comment on the usage of complex sentence forms, including any punctuation or word order issues. Offer specific examples from the candidate's response to illustrate your points.
Additionally, provide actionable suggestions for each criterion to enhance clarity, coherence, vocabulary, and grammatical accuracy.
The graphical image and its description will be given in the Question parameter, and the candidate's response will be in the Answer parameter. 
The overall band score and individual scores for each main criterion in half-band increments (e.g., 7.0, 7.5, 8.0, etc.).
Present the output in the following format:

Overall Band Score: {score}

**{Criterion}:** {score}
- Overall assessment: [Deliver a balanced evaluation of the candidate's performance in relation to the criterion, highlighting strengths and areas for improvement.]
- Detailed explanation: [Thoroughly analyze the candidate's response, underlining its strong points and weaknesses specific to the criterion. Cite exact excerpts from the candidate's writing—sentences, phrases, or words—to exemplify your points. Document recurring error patterns to provide feedback that is closely attuned to the candidate's writing style and needs.]
- Suggestions: [Present comprehensive, criterion-related improvement strategies with examples. For each identified issue, provide an example of how the candidate can make specific enhancements, making your suggestions directly applicable to the examples cited.]
'''

# Keys: Assessment
system_content_writing_task_translation = '''
Act as a Vietnamese translator to convert IELTS writing task examiner assessments given in the Assessment parameter from English to Vietnamese, adhering to these enhanced instructions:
- Keep listed, compared, cited english words, phrases, or examples, and content within double quotes and parentheses.
- Preserve feedback organization and flow in translation.
- Convey the tone of feedback, whether encouraging or critical, in a manner that preserves its educational impact in the Vietnamese context.
'''

# Keys: Question, Answer
system_content_writing_task_2_assessment = '''
You are an experienced IELTS examiner assigned to evaluate a candidate's IELTS Writing Task 2 essay, which may involve presenting opinions, discussing viewpoints, constructing arguments, or outlining problems with solutions. Your critique should align with the IELTS Writing Task 2 rubric, with feedback that is objective and conducive to improvement. Acknowledge the candidate's successful rhetorical and structural strategies, and identify areas for enhancement.
Commence your evaluation with an appraisal of the thesis statement, considering its clarity and how well it establishes the candidate's main argument in response to the essay prompt.
Proceed with a detailed assessment of the following criteria: task response, coherence and cohesion, lexical resource, grammatical range and accuracy. Discuss the effectiveness of the candidate's language in meeting the task requirements and the clarity with which they express their viewpoints. Evaluate the candidate's understanding of the topic and their ability to construct a logical, substantiated argument.
Examine the vocabulary range and accuracy, noting any errors that might affect clarity, and assess the variety and correctness of sentence structures, including grammar and punctuation. Use specific, cited examples from the candidate's essay to support your analysis.
Additionally, provide specific recommendations to enhance the candidate's task response, coherence, vocabulary, and grammatical precision. Ensure your feedback is detailed to assist the candidate in developing their writing skills.
You should include the overall band score and individual scores for each criterion, given in half-band increments (e.g., 7.0, 7.5, 8.0, etc.).
Present your evaluation in the following format:

Overall Band Score: {score}

**{Criterion}:** {score}
- Overall assessment: [Offer a comprehensive and unbiased assessment of the candidate's essay in light of the specific criterion, spotlighting both well-executed elements and areas that need work.]
- Detailed explanation: [Scrutinize the candidate's essay meticulously, highlighting its strong points and shortcomings relevant to the criterion. Cite direct quotes from the candidate's essay—phrases, clauses, or full sentences—to concretize your observations. Uncover patterns in any errors to customize your feedback to the candidate's writing style and challenges.]
- Suggestions: [Articulate exhaustive, criterion-specific improvement suggestions complete with examples. For every problem noted, provide an example of how the candidate can ameliorate their essay, linking your advice directly to the examples cited.]

Thesis Statement Assessment: [Evaluate the thesis statement's effectiveness in presenting the main argument or viewpoint and its responsiveness to the essay prompt.]
'''

# Keys: Question, Answer
system_content_writing_task_1_model_anwser= '''
Your expertise as an IELTS instructor will be crucial for interpreting the visual information presented in the graphic or diagram for Writing Task 1. 
You are given a graphical image file with its description. The description is in the Question parameter and student's answer is in the Answer parameter.
You'll encounter various graphical images such as bar charts, line graphs, pie charts, tables, flow charts, diagrams, and maps. Each type has unique features: bar charts compare groups or track changes, line graphs show trends over time, pie charts represent proportions, tables organize data for comparison, flow charts depict processes, diagrams explain how things work, and maps illustrate geographical data. 
When describing these images, provide an overview of main trends or stages, identify significant points like high and low values, and make relevant comparisons without giving personal opinions.
Focus on summarizing key details accurately and coherently.
You will write a model answer thoroughly and accurately reflects all essential elements of the visual data to achieve higher score than student's answer.
To ensure the model answer reflects a high level of data comprehension, consider the following details:
- Carefully analyze the visual graphic to understand the data set and the story it tells.
- Identify the type of graphical image and adjust your approach to describe its data accordingly.
- Pay attention to the units of measurement and time frames to accurately reference the data.
- When describing the data, use a range of descriptive language and appropriate terminology related to data interpretation.
- Ensure that every statement about the data can be substantiated by the information provided in the graphic or diagram.
- Exercise caution in distinguishing and correlating each label with its corresponding data set. This is crucial as labels can sometimes be lengthy and may overlap, leading to potential confusion.
- Highlight the relationship between these data points and the overarching label in your writing, ensuring that your response captures the full scope of the information presented.
- Adhere to the IELTS Writing Task 1 word count, ensuring your answer is between 150 and 200 words.

Please present your answer in the following format:
"""
{Your Answer}

(Word count: {Number of words in your model answer}) \n
(Overall Band Score: {Band score for your model answer, in half-band increments})
"""
'''

# Keys: Question, Answer
system_content_writing_task_2_model_anwser = '''
As an adept IELTS instructor specialized in Writing Task 2, you are responsible for helping a student improve their essay writing abilities for the IELTS exam. 
The student will present you with an essay topic and their written response. 
Your task is to provide a model essay that not only argues a position with clarity and depth but also reflects the criteria necessary to attain a high band score that surpasses the student's current level.
After reviewing the essay topic given in the Question parameter and the student's essay in the Answer parameter, you are to:
  - Formulate a clear and concise thesis statement that encapsulates the main argument of your model essay.
  - Write a model essay that thoroughly addresses the essay topic, showcasing a well-reasoned and structured argument.
  - Ensure that your essay adheres to the IELTS Writing Task 2 word count requirement, with a minimum of 250 words and a maximum of 300 words.
  - Exhibit an advanced level of proficiency in essay writing, surpassing the student's original submission in terms of argument development, vocabulary range, and grammatical accuracy.
  - Organize your essay with a clear introduction that includes a thesis statement, body paragraphs that each contain a single, coherent idea supported by evidence or examples, and a conclusion that summarizes the main points and restates the thesis.
  - Assign an estimated overall band score to your model essay, reflecting the IELTS scoring criteria in half-band increments (e.g., 6.5, 7.0, 7.5, etc.).

Your model essay, preceded by its thesis statement, should act as a benchmark for the student, illustrating the elevated writing standard necessary to achieve a higher band score.
Please format your model essay as follows:

**Thesis Statement:** {Thesis statement of your model essay}

**Model Essay:** \n
{Your Model Essay Text}

(Word count: {Number of words in your model essay}) \n
(Overall Band Score: {Band score for your model essay, in half-band increments})

It is imperative that your model essay not only adheres to IELTS Writing Task 2 standards regarding format and word limit but also acts as a precise benchmark for the student, illustrating the level of quality required to achieve a high band score.
'''

def claude_create_user_content(question: str, answer: str):
  text = ""
  if question:
    text = f'The Question:\n{json.dumps(question)}'

  if answer:
    text += f'\n\nThe Answer:\n{json.dumps(question)}'
  
  return text

def claude_prompt_for_translate(assessment: str):
  user_content = f'The original English Assessment: \n{assessment}'
  return user_content
  
def create_user_promt(question, answer, image_desc=""):
  user_content = ""
  if question:
    user_content = f'Question:\n"{question}"'

  if answer:
    user_content += f'\n\nThe student answer:\n"{answer}"'
  if image_desc != "":
    user_content += f'\n\nThe image description of the IELTS Writing Task 1 exam:\n"{image_desc}"'
  return user_content

def main():
  st.set_page_config(page_title="Scoring & Assessment of IELTS Writing Task 1 & 2", page_icon="📚", layout="wide")
  st.title('Model Claude_v3 Sonnet')
  # hide menu/footer
  hide_st_style = """
      <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      </style>
  """
  st.markdown(hide_st_style, unsafe_allow_html=True)

  category_options = [
      "Writing Task 1",
      "Writing Task 2",
  ]
  selected_category = st.radio("Please Choose", category_options)

  file = None
  if selected_category == "Writing Task 1":
    st.subheader("Scoring Writing Task 1")
    txt_task1_question = st.text_area("Writing Task1 Question", help="Input the question here", height=100, max_chars=2048, key="txt_task1_question")
    file = st.file_uploader('Wrting Task1 Question', type=["jpg", "png"], accept_multiple_files=False, key="files")
    txt_student_response = st.text_area("The student response", help="Input the student reponse here", height=200, max_chars=4096, key="txt_student_response")
  else:
    st.subheader("Scoring Writing Task 2")
    txt_task2_question = st.text_area("Writing Task2 Question", help="Input the question here", height=100, max_chars=2048, key="txt_task2_question")
    txt_student_response = st.text_area("The student response", help="Input the student reponse here", height=200, max_chars=4096, key="txt_task1_question")

  
  tmp_folder = "tmp"
  if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)
   
  if file is not None:
    # Convert the file to an image data URL
    image = Image.open(file)
    image_path = os.path.join(tmp_folder, file.name)
    image.save(image_path)

    with st.spinner('Studying the image...'):
      image_data = local_image_to_base64(image_path)
      st.image(image, caption='Uploaded Image', use_column_width=True) 
  
  default_lang = st.radio("Default Answer Language", ["English", "Vietnamese"])

  if selected_category == "Writing Task 1":
    if not (txt_task1_question and txt_student_response and file):
      st.warning("Please enter the Question and Student-response and upload the image")
    else:
      scoring_expander = st.expander("Scoring & assessment")
      model_answer_expander = st.expander("Model Answer")
      # Writing Task 1
      if st.button("Scoring"):
        assessment_res = ""
        with scoring_expander:
          st.subheader("Scoring & Assessment")
          message_placeholder = st.empty()
          with st.spinner('Generating...'):
            user_content = claude_create_user_content(
                question=txt_task1_question,
                answer=txt_student_response,
            )
            client = boto3.client(service_name="bedrock-runtime", region_name=REGION)
            wrapper = BedrockRuntimeWrapper(client)
            response = wrapper.invoke_claude_3_multimodal(
                system_content=system_content_writing_task_1_assessment,
                user_content=user_content,
                image_source=image_data
            )
            
            assessment_res = ""
            try:
              output_list = response.get("content", [])
              for output in output_list:
                assessment_res += (output["text"] or "")
                message_placeholder.markdown(assessment_res + "▌")
              message_placeholder.markdown(assessment_res)
            except ClientError:
              logger.exception("Couldn't invoke model with response stream")
              raise
            

        with model_answer_expander:
          st.subheader("Model Answer")
          message_placeholder = st.empty()
          with st.spinner('Generating...'):
            user_content = claude_create_user_content(
                question=txt_task1_question,
                answer=txt_student_response,
            )
            client = boto3.client(service_name="bedrock-runtime", region_name=REGION)
            wrapper = BedrockRuntimeWrapper(client)
            response = wrapper.invoke_claude_3_multimodal(
                system_content=system_content_writing_task_1_model_anwser,
                user_content=user_content,
                image_source=image_data
            )
            full_response = ""
            try:
              output_list = response.get("content", [])
              for output in output_list:
                full_response += (output["text"] or "")
                message_placeholder.markdown(full_response + "▌")
              message_placeholder.markdown(full_response)
            except ClientError:
              logger.exception("Couldn't invoke model with response stream")
              raise

        # Translate the response to Vietnamese
        with scoring_expander:
          if default_lang == "Vietnamese" and assessment_res != "":
            st.subheader("Translate to Vietnamese")
            markdown_trans= st.empty()
            with st.spinner('Generating...'):
              user_content = claude_prompt_for_translate(
                assessment=assessment_res
              )
              wrapper = BedrockRuntimeWrapper()
              response = wrapper.invoke_claude_3_multimodal(
                  system_content=system_content_writing_task_translation,
                  user_content=user_content
              )

              trans_content = ""
              output_list = response.get("content", [])
              for output in output_list:
                trans_content += (output["text"] or "")
                markdown_trans.markdown(trans_content + "▌")
              markdown_trans.markdown(trans_content)
  else:
    scoring_expander = st.expander("Scoring & assessment")
    model_answer_expander = st.expander("Model Answer")
    
    if not (txt_task2_question and txt_student_response):
      st.warning("Please enter the question and answer")
    else:
      # Writing Task 2
      if st.button("Scoring"):
        # Writing task 2 scoring & assessment
        assessment_res = ""
        with scoring_expander:
          st.subheader("Scoring & Assessment")
          message_placeholder = st.empty()
          with st.spinner('Generating...'):
            user_content = claude_create_user_content(
                question=txt_task2_question,
                answer=txt_student_response,
            )
            wrapper = BedrockRuntimeWrapper()
            response = wrapper.invoke_claude_3_multimodal(
                system_content=system_content_writing_task_2_assessment,
                user_content=user_content,
            )
            
            assessment_res = ""
            try:
              output_list = response.get("content", [])
              for output in output_list:
                assessment_res += (output["text"] or "")
                message_placeholder.markdown(assessment_res + "▌")
              message_placeholder.markdown(assessment_res)
            except ClientError:
              logger.exception("Couldn't invoke model with response stream")
              raise


        # Model Answer for Writing Task 2
        with model_answer_expander:
          st.subheader("Model Answer")
          message_placeholder = st.empty()
          with st.spinner('Generating...'):
            user_content = claude_create_user_content(
                question=txt_task2_question,
                answer=txt_student_response,
            )
            wrapper = BedrockRuntimeWrapper()
            response = wrapper.invoke_claude_3_multimodal(
                system_content=system_content_writing_task_2_model_anwser,
                user_content=user_content
            )
            full_response = ""
            try:
              output_list = response.get("content", [])
              for output in output_list:
                full_response += (output["text"] or "")
                message_placeholder.markdown(full_response + "▌")
              message_placeholder.markdown(full_response)
            except ClientError:
              logger.exception("Couldn't invoke model with response stream")
              raise

        # Translate the response to Vietnamese
        with scoring_expander:
          if default_lang == "Vietnamese" and assessment_res != "":
            st.subheader("Translate to Vietnamese")
            markdown_trans= st.empty()
            with st.spinner('Generating...'):
              user_content = claude_prompt_for_translate(
                assessment=assessment_res
              )
              wrapper = BedrockRuntimeWrapper()
              response = wrapper.invoke_claude_3_multimodal(
                  system_content=system_content_writing_task_translation,
                  user_content=user_content
              )

              trans_content = ""
              output_list = response.get("content", [])
              for output in output_list:
                trans_content += (output["text"] or "")
                markdown_trans.markdown(trans_content + "▌")
              markdown_trans.markdown(trans_content)
      
if __name__ == "__main__":
    main()
