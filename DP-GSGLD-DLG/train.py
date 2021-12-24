import time
import torch



def train(model, n_epochs, device, criterion, optimizer, train_data, 
	batch_size_train, param_threshold,
	train_accuracy_history,
    param_history, gradient_norm_history, train_loss_history, time_consume_history, argsopt):


    for epoch in range(1, n_epochs+1):
        train_loss = 0.0 
        valid_loss = 0.0
        train_acc =0.0


        training_sampler  = torch.utils.data.RandomSampler(
            train_data, replacement=True, 
            num_samples=batch_size_train, 
            )

        sampled_training_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size_train, 
            sampler = training_sampler
            )

        model.train()
        time_start = time.time()

        batch_idx, (data, target) = next(
            enumerate(sampled_training_loader)
            )

        #print(data.shape)
        #print('==========target shape========',target.shape)



        data, target = data.to(device), target.to(device) 
        output = model(data)
        #print('============output shape======',output.shape)
        #print('==========target shape========',target.shape)

        loss = criterion(output, target) 


        optimizer.zero_grad()
        loss.backward()

        if argsopt == 'FDPGSGLD':
            optimizer.step(epoch=epoch)
        else:
            optimizer.step()
        #print(optimizer.get_gradient)


        for param in model.parameters():
            #print(param.grad.shape)
            #print(param.grad.norm())
            if argsopt == 'DPGSGLD' or argsopt == 'FDPGSGLD':
                torch.nn.utils.clip_grad_norm(param, param_threshold) 
            else:
                pass
            param_history.append(param)
            gradient_norm_history.append(param.grad.norm().detach().cpu())

        train_loss += loss.item()*data.size(0)      
        pred = output.argmax(dim=1, keepdim=True)
        #print('=====pred========',pred.shape)

        correct = pred.eq(target.view_as(pred)).sum().item()
        train_acc += correct


        train_loss = train_loss / len(data)
        train_accuracy = 100 * train_acc / len(data)


        time_elapsed = time.time() - time_start


        time_consume_history.append(time_elapsed)
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        print(f'| Epoch: {epoch: 02} | Train Loss: {train_loss:.3f} | Time comsume: {time_elapsed:.3f} | Train Accuracy: {train_accuracy: .3f}% |')

    return train_loss_history, time_consume_history, param_history, gradient_norm_history, train_accuracy_history
