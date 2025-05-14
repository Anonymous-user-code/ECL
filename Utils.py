import torch
import torch.nn as nn
from scipy import optimize
import numpy as np
from scipy.optimize import minimize
from sklearn import linear_model
from torch.optim import lr_scheduler
import math
import torch.nn.functional as F
from copy import deepcopy


#-------------------------------------------TransCal--------------------------------------#
def cal_acc_error(logit, label):
    softmaxes = nn.Softmax(dim=1)(logit)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(label)
    accuracy = accuracies.float().mean()
    confidence = confidences.float().mean()
    error = 1 - accuracies.float()
    error = error.view(len(error), 1).float().numpy()
    return accuracy, confidence, error

class TempScaling(nn.Module):
    def __init__(self):
        super(TempScaling, self).__init__()

    def find_best_T(self, logits, labels):
        nll_criterion = nn.CrossEntropyLoss()
        def eval(x):
            "x ==> temperature T"
            x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            loss = torch.mean(nll_criterion(scaled_logits, labels))
            return loss
        optimal_parameter = optimize.fmin(eval, 2.0, disp=False)
        self.temperature = optimal_parameter[0]
        return self.temperature.item()

class TransCal(nn.Module):
    def __init__(self, bias_term=True, variance_term=True):
        super(TransCal, self).__init__()
        self.bias_term = bias_term
        self.variance_term = variance_term

    def find_best_T(self, logits, weight, error, source_confidence):
        def eval(x):
            "x[0] ==> temperature T"
            scaled_logits = logits / x[0]

            "x[1] ==> learnable meta parameter \lambda"
            if self.bias_term:
                controled_weight = weight ** x[1]
            else:
                controled_weight = weight

            ## 1. confidence
            max_L = np.max(scaled_logits, axis=1, keepdims=True)
            exp_L = np.exp(scaled_logits - max_L)
            softmaxes = exp_L / np.sum(exp_L, axis=1, keepdims=True)
            confidences = np.max(softmaxes, axis=1)
            confidence = np.mean(confidences)

            ## 2. accuracy
            if self.variance_term:
                weighted_error = controled_weight * error
                cov_1 = np.cov(np.concatenate((weighted_error, controled_weight), axis=1), rowvar=False)[0][1]
                var_w = np.var(controled_weight, ddof=1)
                eta_1 = - cov_1 / (var_w+1e-8)

                cv_weighted_error = weighted_error + eta_1 * (controled_weight - 1)
                correctness = 1 - error
                cov_2 = np.cov(np.concatenate((cv_weighted_error, correctness), axis=1), rowvar=False)[0][1]
                var_r = np.var(correctness, ddof=1)
                eta_2 = - cov_2 / (var_r)

                target_risk = np.mean(weighted_error) + eta_1 * np.mean(controled_weight) - eta_1 \
                              + eta_2 * np.mean(correctness) - eta_2 * source_confidence
                estimated_acc = 1.0 - target_risk
            else:
                weighted_error = controled_weight * error
                target_risk = np.mean(weighted_error)
                estimated_acc = 1.0 - target_risk

            loss = np.abs(confidence - estimated_acc)
            return loss

        bnds = ((1.0, None), (0.0, 1.0))
        optimal_parameter = minimize(eval, np.array([2.0, 0.5]), method='SLSQP', bounds=bnds)
        self.temperature = optimal_parameter.x[0]
        return self.temperature.item()

def get_weight_union(source_train_feature, target_feature, source_val_feature):
    """
    :param source_train_feature: shape [n_tr, d], features from training set
    :param target_feature: shape [n_t, d], features from test set
    :param source_val_feature: shape [n_v, d], features from validation set

    :return:
    """
    print("-"*30 + "get_weight" + '-'*30)
    n_tr, d = source_train_feature.shape
    n_t, _d = target_feature.shape
    n_v, _d = source_val_feature.shape
    print("n_tr: ", n_tr, "n_v: ", n_v, "n_t: ", n_t, "d: ", d)

    if n_tr < n_t:
        sample_index = np.random.choice(n_tr,  n_t, replace=True)
        source_train_feature = source_train_feature[sample_index]
        sample_num = n_t
    elif n_tr > n_t:
        sample_index = np.random.choice(n_t, n_tr, replace=True)
        target_feature = target_feature[sample_index]
        sample_num = n_tr

    combine_feature = np.concatenate((source_train_feature, target_feature))
    combine_label = np.asarray([1] * sample_num + [0] * sample_num, dtype=np.int32)
    domain_classifier = linear_model.LogisticRegression(max_iter=500,solver="saga")
    domain_classifier.fit(combine_feature, combine_label)
    domain_out = domain_classifier.predict_proba(source_val_feature)
    weight = domain_out[:, :1] / domain_out[:, 1:]
    return weight

def TransCal_Fun(logits_source_val, labels_source_val,logits_target, weight):
    cal_model = TempScaling()
    optimal_temp_source = cal_model.find_best_T(logits_source_val, labels_source_val)
    _, source_confidence, error_source_val = cal_acc_error(logits_source_val / optimal_temp_source, labels_source_val)

    cal_model = TransCal()
    optimal_temp = cal_model.find_best_T(logits_target.numpy(), weight, error_source_val, source_confidence.item())
    return optimal_temp


#-------------------------------------------DRL--------------------------------------#
class ClassificationFunctionAVH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, Y, r_st, bias=True):
        # Normalize the output of the classifier before the last layer using the criterion of AVH
        exp_temp = input.mm(weight.t()).mul(r_st)
        # forward output for confidence regularized training KL version, r is some ratio instead of density ratio
        r = 0.001 # another hyperparameter that need to be tuned
        new_exp_temp = (exp_temp + r*Y)/(r*Y + torch.ones(Y.shape).to(input.device))
        exp_temp = new_exp_temp

        if bias is not None:
            exp_temp += bias.unsqueeze(0).expand_as(exp_temp)
        output = F.softmax(exp_temp, dim=1)
        ctx.save_for_backward(input, weight, bias, output, Y)
        return exp_temp

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, output, Y = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_Y = grad_r = None
        if ctx.needs_input_grad[0]:
            # not negative here, which is different from math derivation
            grad_input = (output - Y).mm(weight)#/(torch.norm(input, 2)*torch.norm(weight, 2))#/(output.shape[0]*output.shape[1])
        if ctx.needs_input_grad[1]:
            grad_weight = ((output.t() - Y.t()).mm(input))#/(torch.norm(input, 2)*torch.norm(weight, 2))#/(output.shape[0]*output.shape[1])
        if ctx.needs_input_grad[2]:
            grad_Y = None
        if ctx.needs_input_grad[3]:
            grad_r = None
        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_Y, grad_r, grad_bias

class ClassifierLayerAVH(nn.Module):
    """
    The last layer for C
    """
    def __init__(self, input_features, output_features, bias=True):
        super(ClassifierLayerAVH, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter("bias", None)

        # Weight initialization
        self.weight.data.uniform_(-1./math.sqrt(input_features), 1./math.sqrt(input_features))
        if bias is not None:
            self.bias.data.uniform_(-1./math.sqrt(input_features), 1./math.sqrt(input_features))

    def forward(self, input, Y, r):
        return ClassificationFunctionAVH.apply(input, self.weight, Y, r, self.bias)

    def extra_repr(self):
        return "in_features={}, output_features={}, bias={}".format(
            self.input_features, self.output_features, self.bias is not None
        )

class alpha_office(nn.Module):
    def __init__(self, n_output,base_model):
        super(alpha_office, self).__init__()
        self.model = base_model
        num_ftrs = self.model.in_features
        extractor = torch.nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs//2),
            nn.ReLU(),
        )
        self.model.classifier = extractor
        self.final_layer = ClassifierLayerAVH(num_ftrs//2, n_output, bias=True)

    def forward(self, x_s, y_s, r):
        x1 = self.model(x_s)
        x = self.final_layer(x1, y_s, r)
        return x

class GradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, nn_output, prediction, p_t, sign_variable):
        ctx.save_for_backward(input, nn_output, prediction, p_t, sign_variable)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, nn_output, prediction, p_t, sign_variable = ctx.saved_tensors
        grad_input = grad_out = grad_pred = grad_p_t = grad_sign = None
        if ctx.needs_input_grad[0]:
            # The parameters here controls the uncertainty measurement entropy of the results
            if sign_variable is None:
                grad_input = grad_output # original *1e2
            else:
                grad_source = torch.sum(nn_output.mul(prediction), dim=1).reshape(-1,1)/p_t
                grad_target = torch.sum(nn_output.mul(prediction), dim=1).reshape(-1,1) * (-(1-p_t)/p_t**2)
                grad_source /= prediction.shape[0]
                grad_target /= prediction.shape[0]
                grad_input = 1e-1 * torch.cat((grad_source, grad_target), dim=1)/p_t.shape[0]
            grad_input = 1e1 * grad_input # original 1e1
        if ctx.needs_input_grad[1]:
            grad_out = None
        if ctx.needs_input_grad[2]:
            grad_pred = None
        if ctx.needs_input_grad[3]:
            grad_p_t = None
        return grad_input, grad_out, grad_pred, grad_p_t, grad_sign

class GradLayer(nn.Module):
    def __init__(self):
        super(GradLayer, self).__init__()

    def forward(self, input, nn_output, prediction, p_t, sign_variable):
        return GradFunction.apply(input, nn_output, prediction, p_t, sign_variable)

    def extra_repr(self):
        return "The Layer After Source Density Estimation"

class beta_office(nn.Module):
    def __init__(self,base_model):
        super(beta_office, self).__init__()
        self.model = base_model
        num_ftrs = self.model.in_features
        fn = torch.nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs//2),
            nn.Tanh(),
            nn.Linear(num_ftrs//2, 2)
        )
        self.model.classifier = fn
        self.grad = GradLayer()

    def forward(self, x, nn_output, prediction, p_t, pass_sign):
        p = self.model(x)
        p = self.grad(p, nn_output, prediction, p_t, pass_sign)
        return p


def train_one_epoch(train_loader, test_loader, model_alpha, model_beta, optimizer_alpha, optimizer_beta, schedular_alpha, schedular_beta, epoch,num_class,DEVICE):
    ## train loader sample number must be smaller than test loader
    model_alpha.train()
    model_beta.train()
    model_alpha = model_alpha.to(DEVICE)
    model_beta = model_beta.to(DEVICE)
    iter_train = iter(train_loader)
    iter_test = iter(test_loader)
    min_len = min(len(train_loader), len(test_loader))
    bce_loss = nn.BCEWithLogitsLoss()
    sign_variable = torch.autograd.Variable(torch.FloatTensor([0]))
    ce_loss = nn.CrossEntropyLoss()
    train_loss, train_acc = 0, 0
    for i in range(min_len):
        input, label = next(iter_train)
        input_test, _ = next(iter_test)
        label_train = label.reshape((-1,))
        label_train = label_train.to(DEVICE)
        input_train = input.to(DEVICE)
        input_test = input_test.to(DEVICE)
        BATCH_SIZE = input.shape[0]
        input_concat = torch.cat([input_train, input_test], dim=0)
        # this parameter used for softlabling
        label_concat = torch.cat(
            (torch.FloatTensor([1, 0]).repeat(input_train.shape[0], 1), torch.FloatTensor([0, 1]).repeat(input_test.shape[0], 1)), dim=0)
        label_concat = label_concat.to(DEVICE)

        prob = model_beta(input_concat, None, None, None, None)
        assert(F.softmax(prob.detach(), dim=1).cpu().numpy().all()>=0 and F.softmax(prob.detach(), dim=1).cpu().numpy().all()<=1)
        loss_dis = bce_loss(prob, label_concat)
        prediction = F.softmax(prob, dim=1).detach()
        p_s = prediction[:, 0].reshape(-1, 1)
        p_t = prediction[:, 1].reshape(-1, 1)
        r = p_s / p_t
        # Separate source sample density ratios from target sample density ratios
        r_source = r[:BATCH_SIZE].reshape(-1, 1)
        r_target = r[BATCH_SIZE:].reshape(-1, 1)
        p_t_source = p_t[:BATCH_SIZE]
        p_t_target = p_t[BATCH_SIZE:]
        label_train_onehot = torch.zeros([BATCH_SIZE, num_class])
        for j in range(BATCH_SIZE):
            label_train_onehot[j][label_train[j].long()] = 1

        theta_out = model_alpha(input_train, label_train_onehot.to(DEVICE), r_source.detach().to(DEVICE))
        source_pred = F.softmax(theta_out, dim=1)
        nn_out = model_alpha(input_test, torch.ones((input_test.shape[0], num_class)).to(DEVICE), r_target.detach().to(DEVICE))

        pred_target = F.softmax(nn_out, dim=1)
        prob_grad_r = model_beta(input_test, nn_out.detach(), pred_target.detach(), p_t_target.detach(),
                                    sign_variable)
        loss_r = torch.sum(prob_grad_r.mul(torch.zeros(prob_grad_r.shape).to(DEVICE)))
        loss_theta = torch.sum(theta_out)

        # Backpropagate
        #if i < 5 and epoch==0:
        if i % 5 == 0:
            optimizer_beta.zero_grad()
            loss_dis.backward()
            optimizer_beta.step()

        #if i < 5 and epoch==0:
        if i % 5 == 0:
            optimizer_beta.zero_grad()
            #loss_r.backward()
            try:
                loss_r.backward(retain_graph=True)
            except RuntimeError as e:
                pass
            optimizer_beta.step()

        if (i + 1) % 1 == 0:
            optimizer_alpha.zero_grad()
            loss_theta.backward()
            optimizer_alpha.step()

        train_loss += float(ce_loss(theta_out.detach(), label_train.long()))
        train_acc += torch.sum(torch.argmax(source_pred.detach(), dim=1) == label_train.long()).float() / BATCH_SIZE
        if i % 10 == 0:
            train_loss = train_loss/(10*BATCH_SIZE)
            train_acc = train_acc/(10)
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Train Loss: {3:.4f} \t Train Acc: {4:.4f}'.format(
                epoch, i, min_len, train_loss, train_acc*100.0))
            train_loss, train_acc = 0, 0
    schedular_alpha.step()
    schedular_beta.step()
    return model_alpha, model_beta, schedular_alpha, schedular_beta

def drl_boost(train_loader, val_loader, base_model, DEVICE,num_class = 10):
    
    model_alpha = alpha_office(num_class,deepcopy(base_model)).to(DEVICE)
    model_beta = beta_office(deepcopy(base_model)).to(DEVICE)
    optimizer_alpha = torch.optim.SGD(model_alpha.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    optimizer_beta = torch.optim.SGD(model_beta.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler_alpha = lr_scheduler.StepLR(optimizer_alpha, step_size=7, gamma=0.1)
    scheduler_beta = lr_scheduler.StepLR(optimizer_beta, step_size=7, gamma=0.1)
    #model_alpha.load_state_dict(torch.load(resume_path), strict=True)
    for epoch in range(20):
        model_alpha, model_beta, scheduler_alpha, scheduler_beta = train_one_epoch(train_loader, val_loader,
                                                                                   model_alpha, model_beta,
                                                                                   optimizer_alpha, optimizer_beta,
                                                                                   scheduler_alpha, scheduler_beta,
                                                                                   epoch,
                                                                                   num_class,
                                                                                   DEVICE)
    return model_alpha,model_beta

#-------------------------------------------PseudoCal--------------------------------------#
class TempScaling(nn.Module):
    def __init__(self):
        super(TempScaling, self).__init__()

    def find_best_T(self, logits, labels):
        nll_criterion = nn.CrossEntropyLoss(reduction='none')
        def eval(x):
            "x ==> temperature T"
            x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            loss = torch.mean(nll_criterion(scaled_logits, labels))
            return loss
        optimal_parameter = optimize.fmin(eval, 2.0, disp=False)
        self.temperature = optimal_parameter[0]
        return self.temperature.item()

def pseudocal(Target_loader, network, DEVICE, class_num = 10):

    ## pseudo-target synthesis
    def mixup(select_loader):
        start_gather = True
        same_cnt = 0
        diff_cnt = 0
        total = 0
        all_diff_idx = None

        with torch.no_grad():
            for ep in range(1):
                for inputs, _ in select_loader:
                    batch_size = inputs.size(0)
                    sample_num = batch_size
                    inputs_a = inputs.to(DEVICE)
                    clb_lam = 0.65
                    rand_idx = torch.randperm(batch_size)
                    inputs_b = inputs_a[rand_idx]
                    outputs_a = network(inputs_a)
                    if type(outputs_a) is tuple:
                        soft_a = outputs_a[1]
                    else:
                        soft_a = outputs_a

                    soft_b = soft_a[rand_idx]
                    same_cnt += (soft_a.max(dim=-1)[1]==soft_b.max(dim=-1)[1]).nonzero(as_tuple=True)[0].shape[0]
                    diff_cnt += (soft_a.max(dim=-1)[1]!=soft_b.max(dim=-1)[1]).nonzero(as_tuple=True)[0].shape[0]
                    
                    ## consider cross-cluster mixup to cover both correct and wrong predictions
                    diff_idx = (soft_a.max(dim=-1)[1]!=soft_b.max(dim=-1)[1]).nonzero(as_tuple=True)[0] + total

                    hard_a = F.one_hot(soft_a.max(dim=-1)[1], num_classes=class_num).float()
                    hard_b = hard_a[rand_idx]

                    mix_inputs = clb_lam * inputs_a + (1 - clb_lam) * inputs_b
                    mix_soft = clb_lam * soft_a.softmax(dim=-1) + (1 - clb_lam) * soft_b.softmax(dim=-1)
                    mix_hard = clb_lam * hard_a + (1 - clb_lam) * hard_b
                    mix_outputs = network(mix_inputs)
                    if type(mix_outputs) is tuple:
                        mix_out = mix_outputs[1]
                    else:
                        mix_out = mix_outputs

                    if start_gather:
                        all_mix_out = mix_out.detach().cpu()
                        all_mix_soft = mix_soft.detach().cpu()
                        all_mix_hard = mix_hard.detach().cpu()
                        all_diff_idx = diff_idx

                        start_gather = False
                    else:
                        all_mix_out = torch.cat((all_mix_out, mix_out.detach().cpu()), 0)
                        all_mix_soft = torch.cat((all_mix_soft, mix_soft.detach().cpu()), 0)
                        all_mix_hard = torch.cat((all_mix_hard, mix_hard.detach().cpu()), 0)
                        all_diff_idx = torch.cat((all_diff_idx, diff_idx), 0)


        all_diff_idx = all_diff_idx.cpu()
        mix_logits = all_mix_out[all_diff_idx]
        mix_soft_labels = all_mix_soft.max(dim=-1)[1][all_diff_idx]
        mix_hard_labels = all_mix_hard.max(dim=-1)[1][all_diff_idx]

        return mix_logits, mix_soft_labels, mix_hard_labels

    def ts(select_loader):
        mix_logits, mix_soft_labels, mix_hard_labels = mixup(select_loader)
        cal_model = TempScaling()
        soft_temp = cal_model.find_best_T(mix_logits, mix_soft_labels)
        hard_temp = cal_model.find_best_T(mix_logits, mix_hard_labels)
        return soft_temp, hard_temp

    soft_t, hard_t = ts(Target_loader)

    return soft_t, hard_t
