from transformers import AutoModel, AutoTokenizer

model_name = "mixedbread-ai/mxbai-embed-large-v1"

# Downloading the model and tokenizer
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Saving the model and tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./tokenizer")
