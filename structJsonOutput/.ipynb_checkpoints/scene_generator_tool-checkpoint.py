from pydantic import BaseModel
from typing import List
from json_tools import extract_json,validate_json_with_model,model_to_json,json_to_pydantic
from gemeni_generate import generate_text

# Define your Pydantic model
class TitlesModel(BaseModel):
    # Define your fields here, for example:
    titles: List[str]

#varaibles
topic = "AI and SEO"

base_prompt = f"Generate 5 Titles for a blog post about the following topic: [{topic}]"

json_model = model_to_json(TitlesModel(titles=['title1', 'title2']))

optimized_prompt = base_prompt + f'.Please provide a response in a structured JSON format that matches the following model: {json_model}'


# Generate content using the modified prompt
gemeni_response = generate_text(optimized_prompt)

# Extract and validate the JSON from the LLM's response
json_objects = extract_json(gemeni_response)

#validate the response
validated, errors = validate_json_with_model(TitlesModel, json_objects)

if errors:
    # Handle errors (e.g., log them, raise exception, etc.)
    print("Validation errors occurred:", errors)

else:
    model_object = json_to_pydantic(TitlesModel, json_objects[0])
    #play with json
    for title in model_object.titles:
        print(title)



