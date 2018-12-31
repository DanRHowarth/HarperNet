## add in early stopping

import copy
import time
import pandas as pd

model_dir = 'saved_models/'

def train_model(n_epochs, train_loader, val_loader, model, optimizer, criterion, print_batch=False, print_epoch=True, model_name='model_1.pt'):
    
    '''
    model_name(str): must end in .pt . Will be saved to defined model_dir
    '''

    # record length of time 
    since_total = time.time()
    
    history = []
    
    # loop over the dataset multiple times
    for epoch in range(n_epochs):  
        
        # initiate a running loss total for train and validation sets 
        running_loss = 0.0
        val_running_loss = 0.0

        # initiate a running accuracy total for train and validation sets
        running_accuracy = 0.0
        val_running_accuracy = 0.0

        # inititate a best accuracy variable and a best model weights dictions
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        print('Epoch {}/{}:'.format(epoch + 1, n_epochs))
        print('-' * 10)
        
        for batch_i, (inputs, labels) in enumerate(train_loader):
            
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            
            # do we need to do thi? prepare the net for training
            model.train()      

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # forward pass to get outputs
            outputs = model(inputs)
            
            ## ACCURACY
            # get predictions to help determine accuracy
            _, predictions = torch.max(outputs, 1)
            
            # get the correction predictions total by comparing predicted with actual
            correct_predictions = torch.sum(predictions == labels).item()
            
            # get an accuracy per batch
            acc_per_batch = correct_predictions / train_loader.batch_size
            
            # calculate a running total of accuracy
            running_accuracy += correct_predictions
            
            # and get an average by dividing this by the size of the dataset
            running_acc_avg = running_accuracy / (train_loader.batch_size * (batch_i + 1))

            ## LOSS
            # calculate the loss 
            loss = criterion(outputs, labels)

            # backward pass to calculate the parameter gradients
            loss.backward()
            
            # store the loss value as a oython number in a variable 
            loss_per_batch = loss.item()
            
            # update the parameters
            optimizer.step()

            # keep a running total of our losses 
            running_loss += loss_per_batch
            
            # and an average per batch 
            running_loss_avg = running_loss / float (batch_i + 1)
            
            if print_batch:
                print('Batch {}: Loss: {:.4f}; Accuracy: {:.4f} '.format(batch_i + 1, loss_per_batch, acc_per_batch))
                
        if print_epoch:        
            print('Loss: {:.4f}; Accuracy: {:.4f}'.format(running_loss_avg, running_acc_avg))  
    
        for batch_ii, (val_inputs, val_labels) in enumerate(val_loader):
            
            if train_on_gpu:
                val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
            
            # no requirement to monitor gradients - LOOK UP
            with torch.no_grad():
                # so set to eval mode - LOOK UP
                model.eval()
    
                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # forward pass to get outputs
                val_outputs = model(val_inputs)

                ## ACCURACY
                # get predictions to help determine accuracy
                _, val_predictions = torch.max(val_outputs, 1)

                # get the correction predictions total by comparing predicted with actual
                val_correct_predictions = torch.sum(val_predictions == val_labels).item()
                
                # val_correct_predictions = val_predictions.eq(val_labels.data.view_as(val_predictions)).item()

                # get an accuracy per batch
                val_acc_per_batch = val_correct_predictions / val_loader.batch_size

                # calculate a running total of accuracy
                val_running_accuracy += val_correct_predictions

                # and get an average by dividing this by the size of the dataset
                val_running_acc_avg = val_running_accuracy / (val_loader.batch_size * (batch_ii + 1))

                ## LOSS
                # calculate the loss  - we don't need to calculate the loss.backward or optimizer step
                val_loss = criterion(val_outputs, val_labels)

                # store the loss value as a oython number in a variable 
                val_loss_per_batch = val_loss.item()

                # keep a running total of our losses 
                val_running_loss += val_loss_per_batch

                # and an average per batch 
                val_running_loss_avg = val_running_loss / float (batch_ii + 1)
                
                if print_batch:
                    print('VAL: Batch {}: Loss: {:.4f}; Accuracy: {:.4f} '
                          .format(batch_ii + 1, val_loss_per_batch, val_acc_per_batch))

            if val_running_acc_avg > best_acc:
                best_acc = val_running_acc_avg
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_dir + model_name)
                # print('Model weights updated')
                    
        if print_epoch:        
            print('VAL: Loss: {:.4f}; Accuracy: {:.4f}'.format(val_running_loss_avg, val_running_acc_avg)) 
            print()
            
        history.append([running_loss_avg, val_running_loss_avg, running_acc_avg, val_running_acc_avg])

    time_total = time.time() - since_total
    print('Training complete in {:.0f}m {:.0f}s'.format(time_total // 60, time_total % 60))
    print('Finished Training')
    
     # load best model weights
    model.load_state_dict(best_model_wts)
    print('Best model weights saved')
    
    history = pd.DataFrame(data=history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    
    return model, history