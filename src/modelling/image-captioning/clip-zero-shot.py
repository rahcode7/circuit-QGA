from PIL import Image
import requests

# url = 'https://rahcode7.github.io/docs/assets/images/1761_d1.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

#image_path = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/train/images/autockt_-644_png.rf.edb5802f3e0958e3f88d94e8fb915937.jpg"
#image_path = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/train/images/autockt_-323_png.rf.a0afbaa7e18cd9ddb05aff68ecbcb38a.jpg"
image_path = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/train/images/autockt_-628_png.rf.6a9257d8efaf53a55429a2e1e772fdd6.jpg"
image = Image.open(image_path)
# #image.save("cats.png")
# #print(image)

# print("imported image")

from transformers import CLIPProcessor, CLIPModel

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


#inputs = processor(text=["five resistors in the photo","four resistors in the photo","three resistors in the photo","two resistors in the photo", "one resistor in the photo"], images=image, return_tensors="pt", padding=True)
inputs = processor(text=["four ammeter in the photo","three ammeter in the photo","two ammeter in the photo", "one ammeter in the photo"], images=image, return_tensors="pt", padding=True)

#inputs = processor(text=["five capacitors","four capacitors","three capacitors","two capacitors", "one capacitor"], images=image, return_tensors="pt", padding=True)

#inputs = processor(text=["five resistors","four resistors","three resistors","two resistors", "one resistor "], images=image, return_tensors="pt", padding=True)
#inputs = processor(text=["5 resistors","4 resistors","3 resistors","2 resistors", "1 resistors"], images=image, return_tensors="pt", padding=True)


outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)


#
#zero_shot = pipeline("zero-shot-image-classification")
#zero_shot(images=image, candidate_labels=["two cats sitting on a couch", "one cat sitting on a couch"])