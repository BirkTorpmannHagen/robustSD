import torch
import torch.nn.functional as F

def odin_fgsm(model, image):
    image.requires_grad = True

    output = model(image)
    if isinstance(output, list):
        output = output[1]  #for njord
    nnOutputs = output
    nnOutputs = nnOutputs
    nnOutputs = nnOutputs[0]
    nnOutputs = nnOutputs - torch.max(nnOutputs)
    nnOutputs = torch.exp(nnOutputs) / torch.sum(torch.exp(nnOutputs))
    nnOutputs = nnOutputs.unsqueeze(0)
    maxIndexTemp = torch.argmax(nnOutputs)
    print(output.shape)
    print(nnOutputs.shape)

    loss = torch.nn.CrossEntropyLoss()(nnOutputs, torch.autograd.Variable(torch.LongTensor([maxIndexTemp]).cuda()))
    loss.backward()
    data_grad = image.grad.data
    data_grad = data_grad.squeeze(0)
    # perturb image
    perturbed_image = image + data_grad.sign() * 0.1
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
def odin(model, image, feature_transform):
    perturbed_image = odin_fgsm(model, image)
    return cross_entropy(model, perturbed_image)


def jacobian(model, image, num_features=1):
    return torch.norm(
        torch.autograd.functional.jacobian(model, image),"fro").item()

def cross_entropy(model, image, num_features=1):
    out = model(image)
    if isinstance(out, list):
        out = out[1]  #for njord
    return F.cross_entropy(out, torch.ones_like(out)).item()


def grad_magnitude(model, image, num_features=1):
    image.requires_grad = True
    output = model(image)
    if isinstance(output, list):
        output = output[1]  #for njord
    loss = torch.nn.CrossEntropyLoss()(output, torch.ones_like(output))
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    return torch.norm(data_grad, "fro"  ).item()

def jjtsvd(model, image, num_features=1):
    jac = torch.autograd.functional.jacobian(model, image)
    jac = jac.flatten(-3).squeeze()
    jj = jac@jac.T
    return torch.svd(jj)[1][0].item()

def jtjsvd(model, image, num_features=1):
    jac = torch.autograd.functional.jacobian(model, image)
    jac = jac.flatten(-3).squeeze()
    jj = jac.T@jac
    return torch.svd(jj)[1][0].item()

def jjmag(model, image, num_features=1):
    jac = torch.autograd.functional.jacobian(model, image)
    jac = jac.flatten(-3).squeeze()
    jj = jac.T@jac
    return torch.norm(jj, "fro").item()

def typicality(model, image, num_features=1):
    assert num_features==1
    return model.estimate_log_likelihood(image)


def condition_number(model, image, num_features=1):
    """
    https://books.google.no/books?id=JaPtxOytY7kC&q=978-0898713619&redir_esc=y#v=onepage&q=978-0898713619&f=false

    :return:
    """
    return torch.norm(
        torch.autograd.functional.jacobian(model, image),"fro").item() / (
        torch.norm(model(image), "fro").item()*torch.norm(image, "fro").item())

if __name__ == '__main__':
    from classifier.resnetclassifier import ResNetClassifier