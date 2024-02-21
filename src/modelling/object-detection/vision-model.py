from roboflow import Roboflow

rf = Roboflow(api_key="xQbOkJIaW7bKCU5vyobC")
project = rf.workspace().project("circuit-recognition")
model = project.version(2).model

sample_img = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/Circuit Recognition.v9i.darknet/valid/259_png.rf.d887e4776d9e61aff1ed34b682eb8c6e.jpg"


# infer on a local image
print(model.predict(sample_img,confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
