import flask
from flask import request
from PIL import Image
from torch import nn
import torchvision.transforms as T
import torch
import numpy as np

app = flask.Flask(__name__)


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(4, 4),
            stride=(1, 1),
            padding=(0, 0),
        )

        self.fully_connected1 = nn.Linear(in_features=120, out_features=84)
        self.fully_connected2 = nn.Linear(in_features=84, out_features=10)

        self.pooling_layer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Convolution Layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling_layer(x)

        # Convolution Layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling_layer(x)
        x = self.dropout(x)

        # Convolution Layer 3
        x = self.conv3(x)
        x = self.relu(x)

        # flatten x
        x = x.view(-1, 120)

        # Fully connected layer 1
        x = self.fully_connected1(x)
        x = self.relu(x)

        # Fully connected layer 2
        x = self.fully_connected2(x)

        return x


@app.route("/image", methods=["POST"])
def handle_request():
    if request.method == "POST":
        imagefile = flask.request.files["image"]
        img = Image.open(imagefile).convert("L")  # load with Pillow
        img = np.array(img)
        transform = T.Compose([T.ToTensor(), T.Resize((28, 28))])
        img = transform(img)
        model = Network()
        model.load_state_dict(
            torch.load("mnist_new.pth", map_location=torch.device("cpu"))
        )
        model.eval()
        results = model(img)
        print(results)
        category = torch.argmax(results)
        print(category)
        category = str(category.item())
        print(category)
        return category


app.run(host="0.0.0.0", port=88, debug=True)
