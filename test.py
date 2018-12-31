def test_model(test_loader, model, optimizer, criterion, print_batch=False):
    '''
    write function here
    '''
    
    print("Testing")
    print('-' * 8)
    
    # initiate a running loss total for test scores
    test_running_loss = 0.0
        
    # initiate a running accuracy total for test scores
    test_running_accuracy = 0.0
    
    for batch, (test_inputs, test_labels) in enumerate(test_loader):
        
        if train_on_gpu:
            test_inputs, test_labels = test_inputs.cuda(), test_labels.cuda()

        # no requirement to monitor gradients - LOOK UP
        with torch.no_grad():
            # so set to eval mode - LOOK UP
            model.eval()

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # forward pass to get outputs
            test_outputs = model(test_inputs)

            ## ACCURACY
            # get predictions to help determine accuracy
            _, test_predictions = torch.max(test_outputs, 1)

            # get the correction predictions total by comparing predicted with actual
            test_correct_predictions = torch.sum(test_predictions == test_labels).item()

            # get an accuracy per batch
            test_acc_per_batch = test_correct_predictions / test_loader.batch_size

            # calculate a running total of accuracy
            test_running_accuracy += test_correct_predictions

            # and get an average by dividing this by the size of the dataset
            test_running_acc_avg = test_running_accuracy / (test_loader.batch_size * (batch + 1))

            ## LOSS
            # calculate the loss  - we don't need to calculate the loss.backward or optimizer step
            test_loss = criterion(test_outputs, test_labels)

            # store the loss value as a oython number in a variable 
            test_loss_per_batch = test_loss.item()

            # keep a running total of our losses 
            test_running_loss += test_loss_per_batch

            # and an average per batch 
            test_running_loss_avg = test_running_loss / float (batch + 1)

            if print_batch:
                print('Batch {}: Loss: {:.4f}; Accuracy: {:.4f} '
                      .format(batch + 1, test_loss_per_batch, test_acc_per_batch))

    print('Loss: {:.4f}; Accuracy: {:.4f}'.format(test_running_loss_avg, test_running_acc_avg)) 


def prediction_actual(sample, transform, model, optimizer):
    '''
    takes in test_data output and returns an image with its correct class and model prediction
    '''
    
    # extract key information as variables
    image, category, label = sample
    
    # transform the image
    data = transform(image)
    
    with torch.no_grad():
    # so set to eval mode - LOOK UP
        model.eval()

        # zero the parameter (weight) gradients
        optimizer.zero_grad()

        data = data.unsqueeze_(0)

        # forward pass to get outputs
        output = model(data)

        ## ACCURACY
        # get predictions to help determine accuracy
        _, test_prediction = torch.max(output, 1)

        prediction = test_prediction.item()
        
        for cat, number in class_mapping:
            if prediction == number:
                convert = cat
    
    plt.imshow(image)
    plt.title('Actual: {} | Predicted: {}'.format(category, convert))
    plt.axis('off')
    