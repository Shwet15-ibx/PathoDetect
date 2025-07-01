class EnsembleModel:
    def __init__(self, resnet_model, efficientnet_model, weights=(0.6, 0.4)):
        self.resnet = resnet_model
        self.efficientnet = efficientnet_model
        self.weights = weights

    def predict(self, patch_img):
        prob_resnet = self.resnet.predict(patch_img)
        prob_efficientnet = self.efficientnet.predict(patch_img)
        prob = self.weights[0] * prob_resnet + self.weights[1] * prob_efficientnet
        return prob 