"""Builds samples/study_sample.pdf, a short original study guide used by the public demo."""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap

PAGES = [
    ("Machine Learning Basics: Study Guide", [
        "Machine learning is a way of building programs that learn patterns from data instead of",
        "following hand-written rules. A model is trained on examples, then used to make predictions",
        "on new data it has not seen before.",
        "",
        "Supervised learning uses labeled examples. Each example has inputs (features) and a known",
        "answer (label). Classification predicts a category, such as spam or not spam. Regression",
        "predicts a number, such as the price of a house.",
        "",
        "Unsupervised learning has no labels. The model looks for structure by itself. Clustering",
        "groups similar items together, for example customers with similar buying habits.",
    ]),
    ("Training, Testing and Overfitting", [
        "Data is split into a training set and a test set. The model learns from the training set.",
        "The test set is kept aside to measure how well the model works on new data.",
        "",
        "Overfitting happens when a model memorizes the training data, including its noise, and then",
        "performs poorly on new data. Underfitting is the opposite: the model is too simple to",
        "capture the pattern, so it does poorly on both training and test data.",
        "",
        "Common ways to reduce overfitting are to collect more data, use a simpler model, stop",
        "training early, or add regularization, which penalizes overly complex models.",
        "Cross-validation repeats the train and test split several times to give a more reliable score.",
    ]),
    ("Measuring a Classifier", [
        "Accuracy is the share of predictions that are correct. It can mislead when one class is rare.",
        "If only 1 percent of emails are fraud, a model that always says not fraud is 99 percent accurate",
        "but useless.",
        "",
        "Precision is the share of predicted positives that are truly positive. Recall is the share of",
        "real positives that the model finds. The F1 score combines precision and recall into one number.",
        "",
        "A confusion matrix counts true positives, false positives, true negatives and false negatives.",
        "All of the measures above can be calculated from those four counts.",
    ]),
    ("Neural Networks and Gradient Descent", [
        "A neural network is made of layers of small units called neurons. Each neuron multiplies its",
        "inputs by weights, adds them up, and passes the result through an activation function such as",
        "ReLU. Stacking many layers lets the network learn complex patterns. This is called deep learning.",
        "",
        "Training adjusts the weights to reduce a loss, a number that measures how wrong the predictions",
        "are. Gradient descent does this by moving each weight a small step in the direction that lowers",
        "the loss. The size of the step is the learning rate. A rate that is too large can overshoot the",
        "best weights, and one that is too small makes training very slow.",
        "",
        "Backpropagation is the method used to compute how much each weight contributed to the loss.",
    ]),
]

c = canvas.Canvas("samples/study_sample.pdf", pagesize=letter)
c.setTitle("Machine Learning Basics: Study Guide")
for title, lines in PAGES:
    c.setFont("Helvetica-Bold", 16)
    c.drawString(60, 730, title)
    c.setFont("Helvetica", 11)
    y = 700
    for line in lines:
        c.drawString(60, y, line)
        y -= 18
    c.showPage()
c.save()
