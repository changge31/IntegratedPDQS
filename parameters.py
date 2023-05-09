class parameters:
    n = 2240
    t = 5000  # number of rounds
    m = 100  # number of instants
    embed = 8  # hidden layer
    beta = round(0.9 * n)
    relax = 0.2
    weight = 0.2
    lamb = 0.1
    lamb_rate = 0.005
    model_rate = 0.005   # trainer learning rate
    proportion_of_0 = 0.2
    space = 100
    d_sum = 2  # the cardinality of the private data 15
    d_lp = 2
    d_median = 40
    median_low = 0   #
    median_high = 13    

    train = 1  # the number of rounds of training in one iteration

    integral_space = 100
    model = 'best_model'
    model2 = 'best_bf_model'
    model3 = 'bf_model'

    sum_dataset_path = "Customers_count.csv"
    median_dataset_path = "Customers_median.csv"

    save_path = "/Users/PycharmProjects/monotonic/" + "n=" + str(n) + "m=" + str(m) + "train=" + str(train) \
                + "t=" + str(t) + "b=" + str(beta) + 'lamb=' + str(lamb) + "lamb_rate=" + str(lamb_rate) \
                + "model_rate=" + str(model_rate) + "emded=" + str(embed) + "relax=" + str(relax) + "weight=" \
                + str(weight)
