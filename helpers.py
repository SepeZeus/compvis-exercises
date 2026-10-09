
import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(img, label, classes, index, total_images):
    """This function displays an individual image with its corresponding label."""

    # Setup subplot layout with current image's position within total images
    plt.subplot(1, total_images, index+1)
    
    # Unnormalize the image
    img = img / 2 + 0.5
    
    # Convert the tensor image to a NumPy array for plotting
    npimg = img.numpy()
    
    # Display the image, transposing the array to put channels as the last dimension
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    # Add the corresponding class label on the x-axis
    plt.xlabel(classes[label])
    
    # Remove x-axis tick marks
    plt.xticks([])
    
    # Remove y-axis tick marks
    plt.yticks([])

def visualize_first8(images, labels, classes):
    """This function displays the first eight images from a batch."""

    # If there are more than 8 images, we select only the first 8 to display
    if len(images) > 8:
        images = images[:8]
        labels = labels[:8]

    # Initialize a figure to plot the images
    plt.figure(figsize=(2*len(images), 2))

    # loop over images and labels and plot individually
    for i in range(len(images)):
        imshow(images[i], labels[i], classes, i, len(images))

    plt.show()

def evaluate_model(net, testloader, classes):
    """This function calculates and prints the overall accuracy and per-class accuracy of a trained model."""

    # Initialization of counters
    correct = 0
    total = 0
    
    # We're not training, so we don't need to calculate the gradients
    with torch.no_grad():
        # Loop over all test data
        for data in testloader:
            # Unpack images and corresponding labels from the current batch
            images, labels = data
            
            # Forward pass the images through the network to get outputs
            outputs = net(images)
            
            # Get the index (class) with the highest score in the output
            _, predicted = torch.max(outputs.data, 1)
            
            # Increment total count
            total += labels.size(0)
            
            # Increment correct count if the prediction matches the label
            correct += (predicted == labels).sum().item()

    # Print overall accuracy
    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f} %')

    # Initialize dictionaries to count correct and total predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # Again, we don't need to calculate gradients
    with torch.no_grad():
        # Loop over all test data
        for data in testloader:
            # Unpack images and corresponding labels from the current batch
            images, labels = data
            
            # Forward pass the images through the network to get outputs
            outputs = net(images)
            
            # Get the index (class) with the highest score in the output
            _, predictions = torch.max(outputs, 1)
            
            # Increment the correct and total count for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # Print per-class accuracy
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
