"""

"""

import os.path
import torch
import numpy as np
from scipy import integrate
import scipy.stats as stats
import math
import random
import pandas as pd
import seaborn as sn
from parameters import parameters
from train import train
from train import trainset_to_x
from Monotonic import Monotonic
import matplotlib.pyplot as plt


def generate_data():    

    output = list()
    k = int(parameters.n * parameters.proportion_of_0)  # the number of 0
    for i in range(parameters.m):
        nums = np.ones(parameters.n)
        nums[:k] = 0
        np.random.shuffle(nums)
        output.append(nums)
    data = np.array(output)
    answer = np.sum(data, axis=1)
    return data, answer


def accuracy(network, testset, private_data, true_answer, query, weight, d):
    if query == 'lp':
        testset = testset[:-1]
        allocation = network(testset).detach()
        obj_test = (allocation * torch.log((1 + allocation) / (1 - allocation))).sum()
        print("(Testset) Total (expected) privacy: ", obj_test)
        xt = torch.tensor(trainset_to_x(1, parameters.n-1, trainset=testset.numpy()))
        xt = torch.transpose(xt, 1, 2)
        yt0 = network(xt)
        yt = yt0 * torch.log((1 + yt0) / (1 - yt0))
        zt = torch.trapezoid(torch.transpose(yt, 2, 1), torch.transpose(xt, 2, 1))
        exp_payment_test = (testset * allocation * torch.log((1 + allocation) / (1 - allocation))) + zt
        print("(Testset) Total (expected) payment: ", exp_payment_test.sum(1))

    else:
        # get allocation and payment vector
        allocation = network(testset).detach()
        
        obj_test = (allocation * torch.log((1 + allocation) / (1 - allocation))).sum()
        print("(Testset) Total (expected) privacy: ", obj_test)
       
        xt = torch.tensor(trainset_to_x(1, parameters.n, trainset=testset.numpy()))
        xt = torch.transpose(xt, 1, 2)
        yt0 = network(xt)
        yt = yt0 * torch.log((1 + yt0) / (1 - yt0))
        
        zt = torch.trapezoid(torch.transpose(yt, 2, 1), torch.transpose(xt, 2, 1))
        exp_payment_test = (testset * allocation * torch.log((1 + allocation) / (1 - allocation))) + zt
        print("(Testset) Total (expected) payment: ", exp_payment_test.sum(1))

    # perturb the private data
    m = parameters.m
    payments = np.zeros(m)
    privacy = np.zeros(m)
    num_of_selected = np.zeros(m)

    if query == 'lp':
        perturbed_matrix = np.empty(shape=(m, parameters.n-1), dtype=int)  # the perturbed dataset
        selection_matrix = np.empty(shape=(m, parameters.n-1), dtype=bool)

        for i in range(m):
            prob = np.random.rand(parameters.n-1)

            q = allocation.detach().numpy()

            random_data = np.random.randint(parameters.lp_low, parameters.lp_high + 1, size=parameters.n-1)
            selection = prob < q

            perturbed_data = np.where(selection, private_data[i], random_data)

            perturbed_matrix[i] = perturbed_data
            selection_matrix[i] = selection

            # get privacy and payment
            epsilon = np.log((1 + q) / (1 - q))  # actual privacy of each agent
            selected_epsilon = epsilon[selection]
            privacy[i] = sum(selected_epsilon)
            num_of_selected[i] = sum(selection)
            pay = exp_payment_test[0].detach().numpy() / q
            paid = pay[selection]
            payments[i] = sum(paid)

    else:
        perturbed_matrix = np.empty(shape=(m, parameters.n), dtype=int)     # the perturbed dataset
        selection_matrix = np.empty(shape=(m, parameters.n), dtype=bool)

        for i in range(m):
            prob = np.random.rand(parameters.n)

            q = allocation.detach().numpy()

            if query == 'sum':
                random_data = np.random.randint(d, size=parameters.n)
            elif query == 'median':
                random_data = np.random.randint(parameters.median_low, parameters.median_high+1, size=parameters.n)

            selection = prob < q

            perturbed_data = np.where(selection, private_data, random_data)

            perturbed_matrix[i] = perturbed_data
            selection_matrix[i] = selection

            # get privacy and payment
            epsilon = np.log((1 + q) / (1 - q))  # actual privacy of each agent
            selected_epsilon = epsilon[selection]
            privacy[i] = sum(selected_epsilon)
            num_of_selected[i] = sum(selection)
            pay = exp_payment_test[0].detach().numpy() / q
            paid = pay[selection]
            payments[i] = sum(paid)

    # get query answer
    if query == "sum":
        answer = perturbed_matrix.sum(axis=1)
        # print("answer: ", answer)
        avg_privacy = np.mean(privacy)
        var_privacy = np.var(privacy)
        avg_payment = np.mean(payments)
        var_payment = np.var(payments)
        avg_selected = np.mean(num_of_selected)
        var_selected = np.var(num_of_selected)

        abs_relative_err = abs(answer - true_answer) / true_answer

        avg_abs_error = np.mean(abs_relative_err)
        var_abs_error = np.var(abs_relative_err)

        stats = np.array(
            [avg_privacy, var_privacy, avg_payment, var_payment, avg_selected, var_selected, avg_abs_error,
             var_abs_error])

        print("Total privacy: ", privacy)
        print("Avg total privacy: ", avg_privacy)
        print("Total payment: ", payments)
        print("Ave payment: ", avg_payment)
        print("Selected users: ", num_of_selected)
        print("Average selected users: ", avg_selected)
        print("answer_list: ", answer)
        print("true_answer: ", true_answer)
        print("average absolute relative error: ", np.mean(abs_relative_err))
        print("Variance: ", np.var(abs_relative_err))
        return privacy, payments, num_of_selected, answer, stats, abs_relative_err
    elif query == "lp":
        answer = (perturbed_matrix * weight).sum(1) / weight.sum(1)
        avg_privacy = np.mean(privacy)
        var_privacy = np.var(privacy)
        avg_payment = np.mean(payments)
        var_payment = np.var(payments)
        avg_selected = np.mean(num_of_selected)
        var_selected = np.var(num_of_selected)
        error = abs(answer - true_answer)
        avg_error = np.mean(error)
        var_error = np.var(error)

        stats = np.array(
            [avg_privacy, var_privacy, avg_payment, var_payment, avg_selected, var_selected, avg_error,
             var_error])

        print("Total privacy: ", privacy)
        print("Avg total privacy: ", avg_privacy)
        print("Total payment: ", payments)
        print("Ave payment: ", avg_payment)
        print("Selected users: ", num_of_selected)
        print("Average selected users: ", avg_selected)
        print("answer_list: ", answer)
        print("true_answer: ", true_answer)
        print("error: ", error)
        # print("absolute relative error: ", abs(relative_error))
        print("average error: ", avg_error)
        print("Variance: ", var_error)
        return privacy, payments, num_of_selected, answer, stats, error
    else:
        answer = np.median(perturbed_matrix, axis=1)
        avg_privacy = np.mean(privacy)
        var_privacy = np.var(privacy)
        avg_payment = np.mean(payments)
        var_payment = np.var(payments)
        avg_selected = np.mean(num_of_selected)
        var_selected = np.var(num_of_selected)
        error = answer - true_answer
        abs_err = abs(answer - true_answer)
        avg_abs_error = np.mean(abs_err)
        var_abs_error = np.var(abs_err)

        stats = np.array(
            [avg_privacy, var_privacy, avg_payment, var_payment, avg_selected, var_selected, avg_abs_error,
             var_abs_error])

        print("Total privacy: ", privacy)
        print("Avg total privacy: ", avg_privacy)
        print("Total payment: ", payments)
        print("Ave payment: ", avg_payment)
        print("Selected users: ", num_of_selected)
        print("Average selected users: ", avg_selected)
        print("answer_list: ", answer)
        print("true_answer: ", true_answer)
        print("error: ", error)
        print("average relative error: ", sum(error) / parameters.m)
        print("average absolute relative error: ", np.mean(abs_err))
        print("Variance: ", np.var(abs_err))
        return privacy, payments, num_of_selected, answer, stats, abs_err


def GR_sum(theta_matrix, data, true_answer, save_path, d):
    theta = theta_matrix.numpy()

    answer = np.zeros(parameters.m)
    select = np.zeros((parameters.m, parameters.n), dtype=float)
    payment = np.zeros((parameters.m, parameters.n), dtype=float)
    privacy = np.zeros(parameters.m)

    for i in range(parameters.m):
        for j in range(parameters.n):
            if theta[j] / (parameters.n - j - 1) > parameters.beta / (j + 1):
                tmp = parameters.beta / j
                payment[i][:j] = min(tmp, theta[j] / (parameters.n - j))

                epsilon = 1 / (parameters.n - j)
                p = (math.exp(epsilon) - 1) / (1 + math.exp(epsilon))   

                prob = np.random.rand(j)
                other = np.zeros(parameters.n - j, dtype=bool)
                selection = prob < p
                final_selection = np.concatenate((selection, other), axis=0)
                select[i] = final_selection
                random_data = np.random.randint(d, size=parameters.n)
                perturbed_data = np.where(final_selection, data, random_data)

                answer[i] = sum(perturbed_data)
                privacy[i] = sum(final_selection) * epsilon

                break


    avg_privacy = np.mean(privacy)
    print("Total privacy: ", privacy)
    var_privacy = np.var(privacy)
    print("Total payment: ", payment.sum(1))
    avg_payment = np.mean(payment.sum(1))
    var_payment = np.var(payment.sum(1))
    print("Selected: ", select.sum(1))
    avg_selected = np.mean(select.sum(1))
    var_selected = np.var(select.sum(1))
    print("Answer: ", answer)
    print("True answer: ", true_answer)
    error = abs(answer - true_answer) / true_answer
    print("Error: ", error)
    avg_err = np.mean(error)
    var_err = np.var(error)
    print("Avg error: ", avg_err)
    print("Variance: ", var_err)

    stats = np.array([avg_privacy, var_privacy, avg_payment, var_payment, avg_selected, var_selected, avg_err, var_err])

    f_gr = save_path + "/gr.npz"
    np.savez(f_gr, select=select, privacy=privacy, payment=payment, answer=answer, true_answer=true_answer)
    return stats, answer



def GR_median(theta_matrix, data, true_answer, save_path):
    theta = theta_matrix.numpy()

    answer = np.zeros(parameters.m)
    select = np.zeros((parameters.m, parameters.n), dtype=float)
    payment = np.zeros((parameters.m, parameters.n), dtype=float)
    privacy = np.zeros(parameters.m)
    for i in range(parameters.m):
        for j in range(parameters.n):

            if theta[j] / (parameters.n - j - 1) > parameters.beta / (j + 1):
                tmp = parameters.beta / j
                payment[i][:j] = min(tmp, theta[j]/(parameters.n-j))

                epsilon = 1 / (parameters.n-j)
                p = (math.exp(epsilon) - 1) / (1 + math.exp(epsilon))
                prob = np.random.rand(j)
                other = np.zeros((parameters.n-j), dtype=bool)
                selection = prob < p
                final_selection = np.concatenate((selection, other), axis=0)
                select[i] = final_selection
                random_data = np.random.randint(parameters.median_low, parameters.median_high+1, size=parameters.n)
                perturbed_data = np.where(final_selection, data, random_data)
                answer[i] = np.median(perturbed_data)
                privacy[i] = sum(final_selection) * epsilon
                break

    avg_privacy = np.mean(privacy)
    print("Total privacy: ", privacy)
    var_privacy = np.var(privacy)
    print("Total payment: ", payment.sum(1))
    avg_payment = np.mean(payment.sum(1))
    var_payment = np.var(payment.sum(1))
    print("Selected: ", select.sum(1))
    avg_selected = np.mean(select.sum(1))
    var_selected = np.var(select.sum(1))
    print("Answer: ", answer)
    print("True answer: ", true_answer)
    error = abs(answer - true_answer)
    print("Error: ", error)
    avg_err = np.mean(error)
    var_err = np.var(error)
    print("Avg error: ", avg_err)
    print("Variance: ", var_err)
    stats = np.array([avg_privacy, var_privacy, avg_payment, var_payment, avg_selected, var_selected, avg_err, var_err])

    f_gr = save_path + "/gr.npz"
    np.savez(f_gr, select=select, privacy=privacy, payment=payment, answer=answer, true_answer=true_answer)
    return stats, answer


def basic_linear_and_quad(testset: object, private_data: object, true_answer: object, query: object, mode: object, weight: object, d: object) -> object:
    theta = testset.numpy()
    if mode == "linear":
        score = 1 - theta

        # compute expected payments
        exp_payment = np.zeros(parameters.n)
        for i in range(parameters.n):
            bid = theta[i]
            x = np.linspace(bid, 1, parameters.space)
            y = (1 - x) * np.log((2 - x) / x)
            inte = integrate.trapz(y, x)
            exp_payment[i] = bid * (1 - bid) * np.log((2 - bid) / bid) + inte

    elif mode == "qua":
        score = np.fmin(0.99, - theta * theta + 1)

        exp_payment = np.zeros(parameters.n)
        for i in range(parameters.n):
            bid = theta[i]
            x = np.linspace(bid, 1, parameters.space)
            y = (-(x * x) + 1) * np.log((1+(-(x * x) + 1)) / (1-(-(x * x) + 1)))
            inte = integrate.trapz(y, x)
            exp_payment[i] = bid * score[i] * np.log((1 + score[i])/(1 - score[i])) + inte

    # allocation
    select = np.zeros(parameters.n, dtype=float)
    total_exp_payment = 0
    for i in range(parameters.n):
        if total_exp_payment + exp_payment[i] <= parameters.beta:
            total_exp_payment += exp_payment[i]
            select[i] = score[i]
        else:
            break

    # perturb the private data
    m = parameters.m
    payments = np.zeros(m)
    privacy = np.zeros(m)
    num_of_selected = np.zeros(m)

    if query == 'lp':
        perturbed_matrix = np.empty(shape=(m, parameters.n-1), dtype=int)  # the perturbed dataset
        selection_matrix = np.empty(shape=(m, parameters.n-1), dtype=bool)
        for i in range(m):
            prob = np.random.rand(parameters.n-1)
            q = select[:-1]
            random_data = np.random.randint(parameters.lp_low, parameters.lp_high + 1, size=(parameters.n - 1))
            selection = prob < q
            perturbed_data = np.where(selection, private_data[i], random_data)
            perturbed_matrix[i] = perturbed_data
            selection_matrix[i] = selection

            # get privacy and payment
            epsilon = np.log((1 + q) / (1 - q))  # actual privacy of each agent
            selected_epsilon = epsilon[selection]
            privacy[i] = sum(selected_epsilon)
            num_of_selected[i] = sum(selection)
            pay = np.zeros(parameters.n-1)    # expected payments / q
            for j in range(parameters.n-1):
                if q[j] <= 0:
                    pay[j] = 0
                else:
                    pay[j] = exp_payment[j] / q[j]
            paid = pay[selection]
            payments[i] = sum(paid)

    else:
        perturbed_matrix = np.empty(shape=(m, parameters.n), dtype=int)  # the perturbed dataset
        selection_matrix = np.empty(shape=(m, parameters.n), dtype=bool)
        for i in range(m):
            prob = np.random.rand(parameters.n)
            q = select
            if query == "sum":
                random_data = np.random.randint(d, size=parameters.n)
            elif query == "median":
                random_data = np.random.randint(parameters.median_low, parameters.median_high+1, size=parameters.n)
            selection = prob < q    # purchased
            perturbed_data = np.where(selection, private_data, random_data)
            perturbed_matrix[i] = perturbed_data
            selection_matrix[i] = selection

            # get privacy and payment
            epsilon = np.log((1 + q) / (1 - q))  # actual privacy of each agent
            selected_epsilon = epsilon[selection]
            privacy[i] = sum(selected_epsilon)
            num_of_selected[i] = sum(selection)
            pay = np.zeros(parameters.n)    # expected payments / q
            for j in range(parameters.n):
                if q[j] <= 0:
                    pay[j] = 0
                else:
                    pay[j] = exp_payment[j] / q[j]
            paid = pay[selection]
            payments[i] = sum(paid)

    # get query answer
    if query == "sum":
        answer = perturbed_matrix.sum(axis=1)
        avg_privacy = np.mean(privacy)
        var_privacy = np.var(privacy)
        avg_payment = np.mean(payments)
        var_payment = np.var(payments)
        avg_selected = np.mean(num_of_selected)
        var_selected = np.var(num_of_selected)
        abs_relative_err = abs(answer - true_answer) / true_answer
        avg_abs_error = np.mean(abs_relative_err)
        var_abs_error = np.var(abs_relative_err)

        stats = np.array(
            [avg_privacy, var_privacy, avg_payment, var_payment, avg_selected, var_selected, avg_abs_error,
             var_abs_error])

        print("Total privacy: ", privacy)
        print("Avg total privacy: ", avg_privacy)
        print("Total payment: ", payments)
        print("Ave payment: ", avg_payment)
        print("Selected users: ", num_of_selected)
        print("Average selected users: ", avg_selected)
        print("answer_list: ", answer)
        print("true_answer: ", true_answer)
        print("average absolute relative error: ", np.mean(abs_relative_err))
        print("Variance: ", np.var(abs_relative_err))
        return privacy, payments, num_of_selected, answer, stats, abs_relative_err
    elif query == "lp":
        answer = (perturbed_matrix * weight).sum(1) / weight.sum(1)
        # print(perturbed_matrix)
        # print(weight)
        # print(answer)
        avg_privacy = np.mean(privacy)
        var_privacy = np.var(privacy)
        avg_payment = np.mean(payments)
        var_payment = np.var(payments)
        avg_selected = np.mean(num_of_selected)
        var_selected = np.var(num_of_selected)
        error = abs(answer - true_answer)
        avg_error = np.mean(error)
        var_error = np.var(error)

        stats = np.array(
            [avg_privacy, var_privacy, avg_payment, var_payment, avg_selected, var_selected, avg_error, var_error])

        print("Total privacy: ", privacy)
        print("Avg total privacy: ", avg_privacy)
        print("Total payment: ", payments)
        print("Ave payment: ", avg_payment)
        print("Selected users: ", num_of_selected)
        print("Average selected users: ", avg_selected)
        print("answer_list: ", answer)
        print("true_answer: ", true_answer)
        print("absolute error: ", error)
        print("average error: ", avg_error)
        print("Variance: ", var_error)
        return privacy, payments, num_of_selected, answer, stats, error
    else:
        answer = np.median(perturbed_matrix, axis=1)
        avg_privacy = np.mean(privacy)
        var_privacy = np.var(privacy)
        avg_payment = np.mean(payments)
        var_payment = np.var(payments)
        avg_selected = np.mean(num_of_selected)
        var_selected = np.var(num_of_selected)
        error = answer - true_answer
        abs_err = abs(error)
        avg_abs_error = np.mean(abs_err)
        var_abs_error = np.var(abs_err)

        stats = np.array(
            [avg_privacy, var_privacy, avg_payment, var_payment, avg_selected, var_selected, avg_abs_error,
             var_abs_error])

        print("Total privacy: ", privacy)
        print("Avg total privacy: ", avg_privacy)
        print("Total payment: ", payments)
        print("Ave payment: ", avg_payment)
        print("Selected users: ", num_of_selected)
        print("Average selected users: ", avg_selected)
        print("answer_list: ", answer)
        print("true_answer: ", true_answer)
        print("error: ", abs_err)
        print("average relative error: ", sum(error) / parameters.m)
        print("average absolute relative error: ", np.mean(abs_err))
        print("Variance: ", np.var(abs_err))
        return privacy, payments, num_of_selected, answer, stats, abs_err


def basic(testset: object, private_data: object, true_answer: object, query: object, mode: object, weight: object, d: object) -> object:
    theta = testset.numpy()
    if mode == "exp":
        
        tmp = np.random.uniform(0.001, 10, size=parameters.m)
        score = np.fmin(0.99, np.exp(-np.array([tmp]).T * theta))

        # compute expected payments
        exp_payment = np.zeros((parameters.m, parameters.n))
        for i in range(parameters.m):
            for j in range(parameters.n):
                bid = theta[j]
                x = np.linspace(bid, 1, parameters.space)
                y = np.exp(-tmp[i] * x) * np.log((1 + np.exp(-tmp[i] * x))/(1 - np.exp(-tmp[i] * x)))
                inte = integrate.trapz(y, x)
                exp_payment[i][j] = bid * score[i][j] * np.log((1 + score[i][j])/(1 - score[i][j])) + inte

    elif mode == "log":
        tmp = np.random.uniform(0.001, 10, size=parameters.m)

        def get_score(k, valuation):
            s = np.fmin(0.9, -np.log(np.array([tmp]).T * valuation))
            s = np.fmax(0.1, s)
            return s

        score = get_score(tmp, theta)

        # compute expected payments
        exp_payment = np.zeros((parameters.m, parameters.n))
        for i in range(parameters.m):
            for j in range(parameters.n):
                bid = theta[j]
                x = np.linspace(bid, 1, parameters.space)
                y = get_score(tmp[i], x)[0] * np.log((1 + get_score(tmp[i], x)[0])/(1 - get_score(tmp[i], x)[0]))
                inte = integrate.trapz(y, x)
                exp_payment[i][j] = bid * score[i][j] * np.log((1 + score[i][j])/(1 - score[i][j])) + inte


    # allocation
    select = np.zeros((parameters.m, parameters.n), dtype=float)
    for i in range(parameters.m):
        total_exp_payment = 0
        for j in range(parameters.n):
            if total_exp_payment + exp_payment[i][j] <= parameters.beta:
                total_exp_payment += exp_payment[i][j]
                select[i][j] = score[i][j]
            else:
 
                break


    # perturb the private data
    m = parameters.m

    payments = np.zeros(m)
    privacy = np.zeros(m)
    num_of_selected = np.zeros(m)

    if query == 'lp':
        perturbed_matrix = np.empty(shape=(m, parameters.n-1), dtype=int)  # the perturbed dataset
        selection_matrix = np.empty(shape=(m, parameters.n-1), dtype=bool)

        for i in range(m):
            prob = np.random.rand(parameters.n-1)

            q = select[i][:-1]

            random_data = np.random.randint(parameters.lp_low, parameters.lp_high + 1, size=(1, parameters.n-1))

            selection = prob < q  # purchased

            perturbed_data = np.where(selection, private_data[i], random_data)

            perturbed_matrix[i] = perturbed_data
            selection_matrix[i] = selection

            # get privacy and payment
            epsilon = np.log((1 + q) / (1 - q))  # actual privacy of each agent
            selected_epsilon = epsilon[selection]
            privacy[i] = sum(selected_epsilon)
            num_of_selected[i] = sum(selection)
            pay = np.zeros(parameters.n-1)
            for j in range(parameters.n-1):
                if q[j] <= 0:
                    pay[j] = 0
                else:
                    pay[j] = exp_payment[i][j] / q[j]
            paid = pay[selection]

            payments[i] = sum(paid)


    else:
        perturbed_matrix = np.empty(shape=(m, parameters.n), dtype=int)  # the perturbed dataset
        selection_matrix = np.empty(shape=(m, parameters.n), dtype=bool)

        for i in range(m):
            prob = np.random.rand(parameters.n)

            q = select[i]

            if query == 'sum':
                random_data = np.random.randint(d, size=(1, parameters.n))
            elif query == 'median':
                random_data = np.random.randint(parameters.median_low, parameters.median_high+1, size=(1, parameters.n))

            selection = prob < q    # purchased

            perturbed_data = np.where(selection, private_data, random_data)

            perturbed_matrix[i] = perturbed_data
            selection_matrix[i] = selection

            # get privacy and payment
            epsilon = np.log((1 + q) / (1 - q))  # actual privacy of each agent
            selected_epsilon = epsilon[selection]
            privacy[i] = sum(selected_epsilon)
            num_of_selected[i] = sum(selection)
            pay = np.zeros(parameters.n)
            for j in range(parameters.n):
                if q[j] <= 0:
                    pay[j] = 0
                else:
                    pay[j] = exp_payment[i][j] / q[j]
            paid = pay[selection]

            payments[i] = sum(paid)

    # get query answer
    if query == "sum":
        answer = perturbed_matrix.sum(axis=1)
        avg_privacy = np.mean(privacy)
        var_privacy = np.var(privacy)
        avg_payment = np.mean(payments)
        var_payment = np.var(payments)
        avg_selected = np.mean(num_of_selected)
        var_selected = np.var(num_of_selected)
        abs_relative_err = abs(answer - true_answer) / true_answer
        avg_abs_error = np.mean(abs_relative_err)
        var_abs_error = np.var(abs_relative_err)

        stats = np.array(
            [avg_privacy, var_privacy, avg_payment, var_payment, avg_selected, var_selected, avg_abs_error,
             var_abs_error])

        print("Total privacy: ", privacy)
        print("Avg total privacy: ", avg_privacy)
        print("Total payment: ", payments)
        print("Ave payment: ", avg_payment)
        print("Selected users: ", num_of_selected)
        print("Average selected users: ", avg_selected)
        print("answer_list: ", answer)
        print("true_answer: ", true_answer)
        print("average absolute relative error: ", np.mean(abs_relative_err))
        print("Variance: ", np.var(abs_relative_err))
        return privacy, payments, num_of_selected, answer, stats, abs_relative_err
    elif query == "lp":
        answer = (perturbed_matrix * weight).sum(1) / weight.sum(1)
        avg_privacy = np.mean(privacy)
        var_privacy = np.var(privacy)
        avg_payment = np.mean(payments)
        var_payment = np.var(payments)
        avg_selected = np.mean(num_of_selected)
        var_selected = np.var(num_of_selected)
        error = abs(answer - true_answer)
        avg_error = np.mean(error)
        var_error = np.var(error)

        stats = np.array(
            [avg_privacy, var_privacy, avg_payment, var_payment, avg_selected, var_selected, avg_error, var_error])

        print("Total privacy: ", privacy)
        print("Avg total privacy: ", avg_privacy)
        print("Total payment: ", payments)
        print("Ave payment: ", avg_payment)
        print("Selected users: ", num_of_selected)
        print("Average selected users: ", avg_selected)
        print("answer_list: ", answer)
        print("true_answer: ", true_answer)
        print("absolute error: ", error)
        print("average error: ", avg_error)
        print("Variance: ", var_error)
        return privacy, payments, num_of_selected, answer, stats, error
    else:
        answer = np.median(perturbed_matrix, axis=1)
        avg_privacy = np.mean(privacy)
        var_privacy = np.var(privacy)
        avg_payment = np.mean(payments)
        var_payment = np.var(payments)
        avg_selected = np.mean(num_of_selected)
        var_selected = np.var(num_of_selected)
        error = answer - true_answer
        abs_err = abs(error)
        avg_abs_error = np.mean(abs_err)
        var_abs_error = np.var(abs_err)

        stats = np.array(
            [avg_privacy, var_privacy, avg_payment, var_payment, avg_selected, var_selected, avg_abs_error,
             var_abs_error])

        print("Total privacy: ", privacy)
        print("Avg total privacy: ", avg_privacy)
        print("Total payment: ", payments)
        print("Ave payment: ", avg_payment)
        print("Selected users: ", num_of_selected)
        print("Average selected users: ", avg_selected)
        print("answer_list: ", answer)
        print("true_answer: ", true_answer)
        print("error: ", abs_err)
        print("average relative error: ", sum(error) / parameters.m)
        print("average absolute relative error: ", np.mean(abs_err))
        print("Variance: ", np.var(abs_err))
        return privacy, payments, num_of_selected, answer, stats, abs_err


def get_true_answer(data, query, weight):
    if query == "sum":
        return sum(data)
    elif query == "lp":

        data = np.tile(data.T, (parameters.m, 1))
        return (data * weight).sum(1)
    else:
        return np.median(data)


def answer_query(query):

    testset = torch.sort(torch.rand(parameters.n))[0]

    test_data = testset.numpy()

    if query == "sum":
        # record parameters
        save_path = parameters.save_path + "data=" + str(parameters.sum_dataset_path) + "," + str(query) + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            print("File exists.")
        f_para = open(save_path + "/parameters.txt", "w")
        f_para.write("n = " + str(parameters.n) + "\n")
        f_para.write("t = " + str(parameters.t) + "\n")
        f_para.write("m = " + str(parameters.m) + "\n")
        f_para.write("beta = " + str(parameters.beta) + "\n")
        f_para.write("lambda = " + str(parameters.lamb) + "\n")
        f_para.write("lambda rate = " + str(parameters.lamb_rate) + "\n")
        f_para.write("model rate = " + str(parameters.model_rate) + "\n")
        f_para.write("proportion of 0 = " + str(parameters.proportion_of_0) + "\n")
        f_para.write("space = " + str(parameters.space) + "\n")
        f_para.write("d = " + str(parameters.d_sum) + "\n")
        f_para.write("train = " + str(parameters.train) + "\n")
        f_para.write("query = " + str(query) + "\n")
        f_para.write("dataset = " + str(parameters.sum_dataset_path) + "\n")
        f_para.close()
        # load private data
        private_data = pd.read_csv(parameters.sum_dataset_path, header=None)
        private_data = private_data.iloc[:parameters.n].to_numpy()  # only read n rows
        private_data = private_data.flatten()
        np.random.shuffle(private_data)

        pd.DataFrame(test_data).to_csv(save_path + '/testset.csv')

        # train network model
        # network = train()
        network = torch.load(parameters.model, map_location=torch.device('cpu'))
        network2 = torch.load(parameters.model2, map_location=torch.device('cpu'))
        network3 = torch.load(parameters.model3, map_location=torch.device('cpu'))

        # test
        # get true answer
        true_answer = get_true_answer(private_data, query, None)
        print("true answer: ", true_answer)
        df_true_answer = pd.DataFrame(np.array([true_answer]), columns=['True Answer'])

        # test network model
        print("network model: ")
        n_privacy, n_payments, n_num_of_selected, n_answer_list, n_stats, n_err \
            = accuracy(network, testset, private_data, true_answer, query, None, parameters.d_sum)
        np.savez(save_path + "/network_data", n_privacy=n_privacy, n_payments=n_payments,
                 n_num_of_selected=n_num_of_selected,
                 n_answer_list=n_answer_list, true_answer=true_answer)
        df_net = pd.DataFrame(n_stats, columns=["Network"])
        an_net = pd.DataFrame(n_answer_list, columns=["Network"])
        print("")

        print("network model2: ")
        n_privacy2, n_payments2, n_num_of_selected2, n_answer_list2, n_stats2, n_err2 \
            = accuracy(network2, testset, private_data, true_answer, query, None, parameters.d_sum)
        np.savez(save_path + "/network_data2", n_privacy=n_privacy2, n_payments=n_payments2,
                 n_num_of_selected=n_num_of_selected2, n_answer_list=n_answer_list2, true_answer=true_answer)
        df_net2 = pd.DataFrame(n_stats2, columns=["Network2"])
        an_net2 = pd.DataFrame(n_answer_list2, columns=["Network2"])
        print("")

        print("network model3: ")
        n_privacy3, n_payments3, n_num_of_selected3, n_answer_list3, n_stats3, n_err3 \
            = accuracy(network3, testset, private_data, true_answer, query, None, parameters.d_sum)
        np.savez(save_path + "/network_data3", n_privacy=n_privacy3, n_payments=n_payments3,
                 n_num_of_selected=n_num_of_selected3, n_answer_list=n_answer_list3, true_answer=true_answer)
        df_net3 = pd.DataFrame(n_stats3, columns=["Network3"])
        an_net3 = pd.DataFrame(n_answer_list3, columns=["Network3"])
        print("")
        
        # # test basic model
        print("basic linear: ")
        bl_privacy, bl_payments, bl_num_of_selected, bl_answer_list, bl_stats, bl_err \
            = basic_linear_and_quad(testset, private_data, true_answer, query, "linear", None, parameters.d_sum)
        np.savez(save_path + "/basic_linear_data", bl_privacy=bl_privacy, bl_payments=bl_payments,
                 bl_num_of_selected=bl_num_of_selected,
                 bl_answer_list=bl_answer_list, true_answer=true_answer)
        df_bl = pd.DataFrame(bl_stats, columns=["Basic Linear"])
        an_bl = pd.DataFrame(bl_answer_list, columns=["Basic Linear"])
        print("")
        print("basic exp: ")
        be_privacy, be_payments, be_num_of_selected, be_answer_list, be_stats, be_err \
            = basic(testset, private_data, true_answer, query, "exp", None, parameters.d_sum)
        np.savez(save_path + "/basic_exp_data", be_privacy=be_privacy, be_payments=be_payments,
                 be_num_of_selected=be_num_of_selected,
                 be_answer_list=be_answer_list, true_answer=true_answer)
        df_be = pd.DataFrame(be_stats, columns=["Basic Exp"])
        an_be = pd.DataFrame(be_answer_list, columns=["Basic Exp"])
        print("")
        print("basic log")
        blog_privacy, blog_payments, blog_num_of_selected, blog_answer_list, blog_stats, blog_err \
            = basic(testset, private_data, true_answer, query, "log", None, parameters.d_sum)
        np.savez(save_path + "/basic_log_data", blog_privacy=blog_privacy, blog_payments=blog_payments,
                 blog_num_of_selected=blog_num_of_selected,
                 blog_answer_list=blog_answer_list, true_answer=true_answer)
        df_blog = pd.DataFrame(blog_stats, columns=["Basic Log"])
        an_blog = pd.DataFrame(blog_answer_list, columns=["Basic Log"])
        print("")
        print("basic qua")
        bqua_privacy, bqua_payments, bqua_num_of_selected, bqua_answer_list, bqua_stats, bqua_err \
            = basic_linear_and_quad(testset, private_data, true_answer, query, "qua", None, parameters.d_sum)
        np.savez(save_path + "/basic_qua_data", blog_privacy=bqua_privacy, blog_payments=bqua_payments,
                 blog_num_of_selected=bqua_num_of_selected,
                 blog_answer_list=bqua_answer_list, true_answer=true_answer)
        df_bqua = pd.DataFrame(bqua_stats, columns=["Basic Qua"])
        an_bqua = pd.DataFrame(bqua_answer_list, columns=["Basic Qua"])
        print("")
        # test baseline model
        print("Ghosh and Roth: ")
        gr_stats, gr_answer = GR_sum(testset, private_data, true_answer, save_path, parameters.d_sum)
        df_gr = pd.DataFrame(gr_stats, columns=["G&R"])
        an_gr = pd.DataFrame(gr_answer, columns=["G&R"])

        # get and store results
        final = pd.concat([df_net, df_net2, df_net3, df_bl, df_be, df_blog, df_bqua, df_gr], axis=1)
        final.to_csv(save_path + '/final.csv')


        answer_list = pd.concat([an_net, an_net2, an_net3, an_bl, an_be, an_blog, an_bqua, an_gr, df_true_answer],
                                axis=1)
        answer_list.to_csv(save_path + '/answers.csv')

    elif query == 'median':
        # record parameters
        save_path = parameters.save_path + "data=" + str(parameters.median_dataset_path) + "," + str(query) + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            print("File exists.")
        f_para = open(save_path + "/parameters.txt", "w")
        f_para.write("n = " + str(parameters.n) + "\n")
        f_para.write("t = " + str(parameters.t) + "\n")
        f_para.write("m = " + str(parameters.m) + "\n")
        f_para.write("beta = " + str(parameters.beta) + "\n")
        f_para.write("lambda = " + str(parameters.lamb) + "\n")
        f_para.write("lambda rate = " + str(parameters.lamb_rate) + "\n")
        f_para.write("model rate = " + str(parameters.model_rate) + "\n")
        f_para.write("proportion of 0 = " + str(parameters.proportion_of_0) + "\n")
        f_para.write("space = " + str(parameters.space) + "\n")
        f_para.write("d = " + str(parameters.d_median) + "\n")
        f_para.write("train = " + str(parameters.train) + "\n")
        f_para.write("query = " + str(query) + "\n")
        f_para.write("dataset = " + str(parameters.median_dataset_path) + "\n")
        f_para.close()
        # load private data
        private_data = pd.read_csv(parameters.median_dataset_path, header=None)
        private_data = private_data.iloc[:parameters.n].to_numpy()  # only read n rows
        private_data = private_data.flatten()
        np.random.shuffle(private_data)

        # generate testset

        pd.DataFrame(test_data).to_csv(save_path + '/testset.csv')

        # train network model
        # network = train()
        network = torch.load(parameters.model, map_location=torch.device('cpu'))
        network2 = torch.load(parameters.model2, map_location=torch.device('cpu'))
        network3 = torch.load(parameters.model3, map_location=torch.device('cpu'))

        # test
        # get true answer
        true_answer = get_true_answer(private_data, query, None)
        print("true answer: ", true_answer)
        df_true_answer = pd.DataFrame(np.array([true_answer]), columns=['True Answer'])

        # test network model
        print("network model: ")
        n_privacy, n_payments, n_num_of_selected, n_answer_list, n_stats, n_err \
            = accuracy(network, testset, private_data, true_answer, query, None, parameters.d_median)
        np.savez(save_path + "/network_data", n_privacy=n_privacy, n_payments=n_payments,
                 n_num_of_selected=n_num_of_selected, n_answer_list=n_answer_list, true_answer=true_answer)
        df_net = pd.DataFrame(n_stats, columns=["Network"])
        an_net = pd.DataFrame(n_answer_list, columns=["Network"])
        print("")

        print("network model2: ")
        n_privacy2, n_payments2, n_num_of_selected2, n_answer_list2, n_stats2, n_err2 \
            = accuracy(network2, testset, private_data, true_answer, query, None, parameters.d_median)
        np.savez(save_path + "/network_data2", n_privacy=n_privacy2, n_payments=n_payments2,
                 n_num_of_selected=n_num_of_selected2, n_answer_list=n_answer_list2, true_answer=true_answer)
        df_net2 = pd.DataFrame(n_stats2, columns=["Network2"])
        an_net2 = pd.DataFrame(n_answer_list2, columns=["Network2"])
        print("")

        print("network model3: ")
        n_privacy3, n_payments3, n_num_of_selected3, n_answer_list3, n_stats3, n_err3 \
            = accuracy(network3, testset, private_data, true_answer, query, None, parameters.d_median)
        np.savez(save_path + "/network_data3", n_privacy=n_privacy3, n_payments=n_payments3,
                 n_num_of_selected=n_num_of_selected3, n_answer_list=n_answer_list3, true_answer=true_answer)
        df_net3 = pd.DataFrame(n_stats3, columns=["Network3"])
        an_net3 = pd.DataFrame(n_answer_list3, columns=["Network3"])
        print("")

        # test basic model
        print("basic linear: ")
        bl_privacy, bl_payments, bl_num_of_selected, bl_answer_list, bl_stats, bl_err \
            = basic_linear_and_quad(testset, private_data, true_answer, query, "linear", None, parameters.d_median)
        np.savez(save_path + "/basic_linear_data", bl_privacy=bl_privacy, bl_payments=bl_payments,
                 bl_num_of_selected=bl_num_of_selected,
                 bl_answer_list=bl_answer_list, true_answer=true_answer)
        df_bl = pd.DataFrame(bl_stats, columns=["Basic Linear"])
        an_bl = pd.DataFrame(bl_answer_list, columns=["Basic Linear"])
        print("")
        print("basic exp: ")
        be_privacy, be_payments, be_num_of_selected, be_answer_list, be_stats, be_err \
            = basic(testset, private_data, true_answer, query, "exp", None, parameters.d_median)
        np.savez(save_path + "/basic_exp_data", be_privacy=be_privacy, be_payments=be_payments,
                 be_num_of_selected=be_num_of_selected,
                 be_answer_list=be_answer_list, true_answer=true_answer)
        df_be = pd.DataFrame(be_stats, columns=["Basic Exp"])
        an_be = pd.DataFrame(be_answer_list, columns=["Basic Exp"])
        print("")
        print("basic log: ")
        blog_privacy, blog_payments, blog_num_of_selected, blog_answer_list, blog_stats, blog_err \
            = basic(testset, private_data, true_answer, query, "log", None, parameters.d_median)
        np.savez(save_path + "/basic_log_data", blog_privacy=blog_privacy, blog_payments=blog_payments,
                 blog_num_of_selected=blog_num_of_selected,
                 blog_answer_list=blog_answer_list, true_answer=true_answer)
        df_blog = pd.DataFrame(blog_stats, columns=["Basic Log"])
        an_blog = pd.DataFrame(blog_answer_list, columns=["Basic Log"])
        print("")
        print("basic qua")
        bqua_privacy, bqua_payments, bqua_num_of_selected, bqua_answer_list, bqua_stats, bqua_err \
            = basic_linear_and_quad(testset, private_data, true_answer, query, "qua", None, parameters.d_median)
        np.savez(save_path + "/basic_qua_data", blog_privacy=bqua_privacy, blog_payments=bqua_payments,
                 blog_num_of_selected=bqua_num_of_selected,
                 blog_answer_list=bqua_answer_list, true_answer=true_answer)
        df_bqua = pd.DataFrame(bqua_stats, columns=["Basic Qua"])
        an_bqua = pd.DataFrame(bqua_answer_list, columns=["Basic Qua"])
        print("")
        # test baseline model
        print("Ghosh and Roth: ")
        gr_stats, gr_answer = GR_median(testset, private_data, true_answer, save_path)
        df_gr = pd.DataFrame(gr_stats, columns=["G&R"])
        an_gr = pd.DataFrame(gr_answer, columns=["G&R"])

        # get and store results
        final = pd.concat([df_net, df_net2, df_net3, df_bl, df_be, df_blog, df_bqua, df_gr], axis=1)
        final.to_csv(save_path + '/final.csv')
        answer_list = pd.concat([an_net, an_net2, an_net3, an_bl, an_be, an_blog, an_bqua, an_gr, df_true_answer], axis=1)
        answer_list.to_csv(save_path + '/answers.csv')


def __main__():
    answer_query("sum")
    answer_query("median")


__main__()

