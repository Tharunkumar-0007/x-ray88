from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import os
import torch
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import openai
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

load_dotenv()

app = Flask(__name__)
CORS(app) 

# Load environment variables
auth_username = os.getenv("AUTH_USERNAME")
auth_password = os.getenv("AUTH_PASSWORD")
cambridgeltl_access_token = os.getenv('CAMBRIDGELTL_ACCESS_TOKEN')
openai_api_key = os.getenv("OPENAI_TOKEN")

# Initialize models and processors
tokenizer = LlamaTokenizer.from_pretrained("cambridgeltl/med-alpaca", token=cambridgeltl_access_token, legacy=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LlamaForCausalLM.from_pretrained(
    "cambridgeltl/med-alpaca",
    token=cambridgeltl_access_token,
    load_in_8bit=True if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else {"": device}
)
model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

model_deplot = Pix2StructForConditionalGeneration.from_pretrained("google/deplot", torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
processor_deplot = Pix2StructProcessor.from_pretrained("google/deplot")

model_med_git = AutoModelForCausalLM.from_pretrained('cambridgeltl/med-git-base', token=cambridgeltl_access_token, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
processor_med_git = AutoProcessor.from_pretrained('cambridgeltl/med-git-base', token=cambridgeltl_access_token)

openai.api_key = openai_api_key

def set_openai_api_key(api_key):
    if api_key and api_key.startswith("") and len(api_key) > 50:
        openai.api_key = api_key

def get_response_from_openai(prompt, model="gpt-3.5-turbo", max_output_tokens=512):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_output_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate(table, question, llm="med-alpaca", **kwargs):
    prompt_input = f"Below is an instruction that describes a task, paired with an input that provides further context of an uploaded image. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Input:\n{table}\n\n### Response:\n"
    prompt_no_input = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n"
    prompt = prompt_input if len(table) > 0 else prompt_no_input
    
    output = "UNKNOWN ERROR"
    if llm == "med-alpaca":
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,  # Specify the number of beams
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=512,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        output = output.split("### Response:")[1].strip()
    elif llm == "gpt-3.5-turbo":
        try:
            output = get_response_from_openai(prompt)
        except Exception as e:
            output = f"Error: {str(e)}"
    else:
        raise RuntimeError(f"No such LLM: {llm}")
        
    return output


def deplot(image, question, llm):
    inputs = processor_deplot(images=image, text="Generate the underlying data table for the figure below:", return_tensors="pt").to(device)
    predictions = model_deplot.generate(**inputs, max_new_tokens=512)
    table = processor_deplot.decode(predictions[0], skip_special_tokens=True).replace("<0x0A>", "\n")
    return table

def med_git(image, question, llm):
    inputs = processor_med_git(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values.to(torch.float16 if device == "cuda" else torch.float32)
    generated_ids = model_med_git.generate(pixel_values=pixel_values, max_length=512)
    captions = processor_med_git.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return captions

def process_document(image, question, llm):
    if image:
        if np.mean(image) >= 128:
            table = deplot(image, question, llm)
        else:
            table = med_git(image, question, llm)
    else:
        table = ""
        
    res = evaluate(table, question, llm=llm)
    return table, res

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            question = request.form.get("question")
            llm = request.form.get("llm", "gpt-3.5-turbo")  # Default to gpt-3.5-turbo
            file = request.files.get("image")

            if file:
                image = Image.open(file)
                table, response = process_document(image, question, llm)
            else:
                table = ""
                response = "No image uploaded."

            return jsonify({'table': table, 'response': response})

        except Exception as e:
            return jsonify({'error': str(e)}), 500  # Return an error message

    return render_template("index.html", table=None, response=None)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
