import torch
import torch.nn as nn
import numpy as np

def cal_loss(loader, model, delta, loss_function):
    """
    Calculate the average loss of a given loader and model.

    Parameters:
    - loader: A given dataloader.
    - model: A given model for which the loss is being calculated.
    - delta: The perturbation parameter.
    - loss_function: Bool value to use a specific loss function or not.

    Returns:
    - loss_total: Average loss of a given loader and model.
    """
    loss_total = 0
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    delta = delta.cuda()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            x_val = data.cuda()
            outputs_ori = model(x_val.cuda())
            _, target_label = torch.max(outputs_ori, 1)
            perturbed = torch.clamp((x_val + delta), 0, 1)
            outputs = model(perturbed)
            if loss_function:
                loss = torch.mean(loss_fn(outputs, target_label))
            else:
                loss = torch.mean(outputs.gather(1, (target_label.cuda()).unsqueeze(1)).squeeze(1))
            loss_total = loss_total + loss
    loss_total = loss_total / (i + 1)
    return loss_total


def uap_sga(model, loader, nb_epoch, eps, beta=9, step_decay=0.1, loss_function=None,
            uap_init=None, batch_size=None, minibatch=10, loader_eval=None, 
            dir_uap=None, center_crop=224, iter=4, momentum=0, img_num=10000):
    """
    Performs the Stochastic-Gradient-Aggregation (SGA) attack.

    Parameters:
    - model: A given model for which the loss is being calculated.
    - loader: A given **training** dataloader.
    - nb_epoch: Number of optimization epochs.
    - eps: Maximum perturbation value (L-infinity) norm.
    - beta: Clamping value.
    - step_decay: Single step size.
    - loss_function: Custom loss function (DEFAULT=None).
    - uap_init: Custom perturbation to start from (DEFAULT=None).
    - batch_size: Batch size (DEFAULT=None).
    - minibatch: Minibatch for SGA (DEFAULT=10).
    - loader_eval: A given **validation** dataloader (DEFAULT=None).
    - dir_uap: Directory of the read/write UAP-based files (DEFAULT=None).
    - center_crop: Image size (DEFAULT=224).
    - iter: Iteration number (K) in the inner iteration (DEFAULT=4).
    - momentum: Momentum item (DEAFAULT=0).
    - img_num: Number of images to perform the attack (DEAFAULT=10000).
    
    Returns:
    - delta.data: SGA-based information.
    - losses: Mean loss values obtained during each epoch of training.
    """
    model.eval()
    np.random.seed(0)
    
    if uap_init is None:
        # Initialize as zero vector
        batch_delta = torch.zeros(batch_size, 3, center_crop, center_crop)  
    else:
        batch_delta = uap_init.unsqueeze(0).repeat([batch_size, 1, 1, 1])

    delta = batch_delta[0]
    losses = []
    
    # Loss function
    if loss_function:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        beta = torch.cuda.FloatTensor([beta])

        def clamped_loss(output, target):
            loss = torch.mean(torch.min(loss_fn(output, target), beta))
            return loss

    batch_delta.requires_grad_()
    v = 0
    for epoch in range(nb_epoch):
        print('Epoch %i/%i' % (epoch + 1, nb_epoch))

        # Perturbation step size with decay
        eps_step = eps * step_decay
        for i, data in enumerate(loader):
            x_val = data
            with torch.no_grad():
                outputs_ori = model(x_val.cuda())
                _, target_label = torch.max(outputs_ori, 1)

            num = x_val.shape[0]
            k = iter
            noise_inner_all = torch.zeros(k * num//minibatch, 3, center_crop, center_crop)
            delta_inner = delta.data

            for j in range(k * num//minibatch):
                label = np.random.choice(num, minibatch, replace=False)

                if j > 0 or i > 0 or epoch>0:
                    batch_delta.grad.data.zero_()
                
                batch_delta.data = delta_inner.unsqueeze(0).repeat([minibatch, 1, 1, 1])
                perturbed = torch.clamp((x_val[label] + batch_delta).cuda(), 0, 1)
                outputs = model(perturbed)

                # Loss function value
                if loss_function:
                    loss = clamped_loss(outputs, target_label[label].cuda())
                else:
                    loss = -torch.mean(outputs.gather(1, (target_label[label].cuda()).unsqueeze(1)).squeeze(1))

                loss.backward()
                grad_inner = batch_delta.grad.data.mean(dim=0)
                delta_inner = delta_inner + grad_inner.sign() * eps_step
                delta_inner = torch.clamp(delta_inner, -eps, eps)
                noise_inner_all[j, :, :, :] = grad_inner
                batch_delta.grad.data.zero_()

            # Batch update with momentum
            if momentum:
                batch_delta_grad = torch.mean(noise_inner_all.detach().clone(), dim=0, keepdim=True).squeeze(0)
                if torch.norm(batch_delta_grad, p=1) == 0:
                    batch_delta_grad = batch_delta_grad
                else:
                    batch_delta_grad = batch_delta_grad / torch.norm(batch_delta_grad, p=1)
                v = 0.9 * v + batch_delta_grad
                grad_sign = v.sign()
            else:
                grad_sign = torch.mean(noise_inner_all.detach().clone(), dim=0, keepdim=True).squeeze(0).sign()

            delta = delta + grad_sign * eps_step
            delta = torch.clamp(delta, -eps, eps)

        loss = cal_loss(loader_eval, model, delta.data, loss_function)
        losses.append(torch.mean(loss.data).cpu())
        
    return delta.data, losses