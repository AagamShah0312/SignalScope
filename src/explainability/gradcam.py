import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()

        output = self.model(input_tensor)

        probabilities = F.softmax(output, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()

        if target_class is None:
            target_class = predicted_class

        score = output[:, target_class]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1)

        # Keep positive contributions
        cam = F.relu(cam)

        # Normalize
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return (
            cam[0].detach().cpu(),
            predicted_class,
            probabilities[0].detach().cpu()
        )

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()