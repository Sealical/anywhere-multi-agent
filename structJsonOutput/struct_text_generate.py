import google.generativeai as genai
from pydantic import BaseModel
from typing import List
from structJsonOutput.json_tools import extract_json,validate_json_with_model,model_to_json,json_to_pydantic
from PIL import Image

# Configure the GEMINI LLM
genai.configure(api_key='YOUR API KEY')
llm_model = genai.GenerativeModel('gemini-pro')
vlm_model = genai.GenerativeModel('gemini-pro-vision')

# define narrator outputs json format
class DescModel(BaseModel):
    desc: str
    object_name: str
    viewpoint: str

# define divergent thinker output format
class SceneModel(BaseModel):
    scenes: List[str]

# define prompt generator output format
class RankModel(BaseModel):
    scenes: List[str]
    rank: List[int]

# define quality checker json format
class QualityModel(BaseModel):
    ques: List[str]
    ans: List[str]

formatModel_map = {"narrator": DescModel,
                   "divergent": SceneModel,
                   "pgenerator":RankModel,
                   "checker":QualityModel}

# VLM narrator setting
narrator_base_prompt = f"You are an analyst and observer, you can give a detailed description of any object and discover the characteristics of that object. Please give a detail description of this image, as well as describing the important features in that image, and then give the name and the viewpoint of this object"
narrator_json_model = model_to_json(formatModel_map['narrator'](desc='object_description', object_name='', viewpoint=''))
narrator_format_prompt = f'. Please provide a response in a structured JSON format that matches the following model: {narrator_json_model}'

# LLM divergent thiker setting
divergent_base_prompt = "You are an expert imaginative photographer who can choose a variety of suitable scenes for any object to take pictures of. I'll follow up by asking you to provide me with scene descriptors, then I'll provide some useful information for your: the object infomation:[{object_name}], the viewpoint: [{viewpoint}] must appear in scene description, and feedback about the object's previous scene result is: [{feedback}]. Please give 5 sets of relevant scene descriptions for this object: [{prompt}]"
divergent_json_model = model_to_json(formatModel_map['divergent'](scenes=['scene_description1','scene_description2']))
divergent_format_prompt = f'. Please provide a response in a structured JSON format that matches the following model: {divergent_json_model}'

# LLM prompt generator setting (add feedback machanism & object info & viewpoint)
pgenerator_base_prompt = "You are an excellent analyst, able to see the correlation between different texts. Now we have a object description: [{img_desc}]. Please give the sort number (from 1 to 5) about these 5 scene description: [{scene_descs}] that most appropriate with the object"
pgenerator_json_model = model_to_json(formatModel_map['pgenerator'](scenes=['','','','',''], rank=[0,1,2,3,4]))
pgenerator_format_prompt = f'. Please provide a response in a structured JSON format that matches the following model: {pgenerator_json_model}'

# VLM quality checker setting
q1 = 'Is it common for [{subject}] to be placed in this context? '
q2 = 'Is [{subject}] placed normally on a platform or on the ground?'
checker_base_prompt = f"You are an analyst expert and an observer of detail. Please give the answer of these questions: [{q1}, {q2}] "
checker_json_model = model_to_json(formatModel_map['checker'](ques=['question1','question2'], ans=['answer1','answer2']))
checker_format_prompt = f'. Please provide a response in a structured JSON format that matches the following model: {checker_json_model}'

#basic generation
def generate_text(prompt):
    if type(prompt)==str:
        response = llm_model.generate_content(prompt)
    else:
        response = vlm_model.generate_content(prompt)
    return response.text

def struct_generate(prompt='', gen_type='narrator'):
    if gen_type=='narrator':
        """
        prompt: img_path:str
        """
        assert type(prompt) == str
        img = Image.open(prompt)
        narrator_prompt = narrator_base_prompt + narrator_format_prompt
        response = generate_text([narrator_prompt, img])
    if gen_type == 'divergent':
        """
        prompt: [text_prompt:str, object_name:str, viewpoint:str, feedback:str]
        """
        assert type(prompt) == list
        text_prompt, object_name, viewpoint, feedback = prompt[0], prompt[1], prompt[2], prompt[3] 
        divergent_prompt = divergent_base_prompt.format(object_name=object_name, viewpoint=viewpoint, feedback=feedback, prompt=text_prompt) + divergent_format_prompt
        response = generate_text(divergent_prompt)
    if gen_type  == 'pgenerator':
        """
        prompt: [scene_description: List[str], image_description: str]
        """
        assert type(prompt) == list
        scene_descs, img_desc= prompt[0], prompt[1]
        pgenerator_prompt = pgenerator_base_prompt.format(scene_descs=scene_descs, img_desc=img_desc)+ pgenerator_format_prompt
        response = generate_text(pgenerator_prompt)
    if gen_type == 'checker':
        """
        prompt: [subject_name: str, img:PIL]
        """
        assert type(prompt) == list
        name, img = prompt[0], prompt[1]
        checker_prompt = checker_base_prompt.format(subject=name) + checker_format_prompt
        response = generate_text([checker_prompt, img])
        
    
    json_objects = extract_json(response)
    print(json_objects,response)
    #validate the response
    validated, errors = validate_json_with_model(formatModel_map[gen_type], json_objects)
    
    if errors:
        # Handle errors (e.g., log them, raise exception, etc.)
        print("Validation errors occurred:", errors)
    
    else:
        model_object = json_to_pydantic(formatModel_map[gen_type], json_objects[0])
    return model_object.dict()


