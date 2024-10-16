from flask import Flask, request, jsonify
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load the configuration and model
config = PeftConfig.from_pretrained("L-NLProc/PredEx_Llama-2-7B_Pred-Exp_Instruction-Tuned")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = PeftModel.from_pretrained(base_model, "L-NLProc/PredEx_Llama-2-7B_Pred-Exp_Instruction-Tuned")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_text = data.get("input", "")
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_length=100)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({"response": decoded_output})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
