import google.generativeai as genai

# Configure the GEMINI LLM
genai.configure(api_key='')
llm_model = genai.GenerativeModel('gemini-pro')
vlm_model = genai.GenerativeModel('gemini-pro-vision')

#basic generation
def generate_text(prompt):
    if type(prompt)==str:
        response = llm_model.generate_content(prompt)
    else:
        response = vlm_model.generate_content(prompt)
    return response.text