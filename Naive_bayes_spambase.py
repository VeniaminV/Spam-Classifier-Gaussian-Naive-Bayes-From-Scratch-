"""
Veniamin Velikoretskikh 
CS445 Machine learning - porgramming assignment 2 
Gaussian Naive Bayes classifier for spambase dataset

The classification task for this dataset is to determine 
whether a given email is spam or not.

The dataset is downloaded automatically from the UCI ML Repository.

Citation:
Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). 
Spambase [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53G6X.
"""

import math
import random
import csv  # for reading the .data file
 
 
# read the data file and split into features (X) and labels (y)
# each row in the file is one email, the last column is 1 (spam) or 0 (not spam)
# everything before the last column is a feature (word frequencies etc.)
def load_data(filename):
    X = []  # features
    y = []  # labels
 
    with open(filename, newline="") as f:
        reader = csv.reader(f)
 
        # check if the first row is a header or real data
        # spambase.data has no header so this should just be data
        first_row = next(reader, None)
        if first_row is not None:
            try:
                values = [float(v) for v in first_row]
                X.append(values[:-1])      # all columns except last are features
                y.append(int(values[-1]))  # last column is the label
            except ValueError:
                pass  # it was a header, skip it
 
        for row in reader:
            if not row:
                continue
            values = [float(v) for v in row]
            X.append(values[:-1])
            y.append(int(values[-1]))
 
    return X, y
 
 
# split data into training and test sets
# the assignment says each set should be about 2300 emails and about 40% spam
# doing a stratified split keeps the spam ratio the same in both halves
# (if we just randomly split everything we might get an uneven ratio by chance)
def stratified_split(X, y, test_fraction=0.5, seed=42):
    rng = random.Random(seed)  # fixed seed so results are the same every run
 
    # separate indices by class so we can split them independently
    idx_not_spam = [i for i, label in enumerate(y) if label == 0]
    idx_spam     = [i for i, label in enumerate(y) if label == 1]
 
    rng.shuffle(idx_not_spam)
    rng.shuffle(idx_spam)
 
    # take the first test_fraction of each class for the test set
    cut_not_spam = int(len(idx_not_spam) * test_fraction)
    cut_spam     = int(len(idx_spam) * test_fraction)
 
    test_idx  = idx_not_spam[:cut_not_spam] + idx_spam[:cut_spam]
    train_idx = idx_not_spam[cut_not_spam:] + idx_spam[cut_spam:]
 
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_test  = [y[i] for i in test_idx]
 
    return X_train, y_train, X_test, y_test
 
 

# mean = average, std = how spread out the values are
def compute_mean_std(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0001
 
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
 
    return mean, std
 
 
# train the naive bayes model
# this just means computing the prior probabilities and the mean/std for each feature
# we store the log of the prior because we'll be adding logs later
# (multiplying 57 tiny probabilities together would underflow to 0.0)
def train_naive_bayes(X_train, y_train):
    n_total    = len(y_train)
    n_features = len(X_train[0])  # should be 57
    classes    = sorted(set(y_train))  # [0, 1]
 
    model = {
        'classes':   classes,
        'log_prior': {},  # log(P(class))
        'means':     {},  # mean of each feature per class
        'stds':      {},  # std of each feature per class
    }
 
    for c in classes:
        # only use the rows that belong to this class
        rows_c = [X_train[i] for i in range(n_total) if y_train[i] == c]
        n_c = len(rows_c)
 
        # prior probability: how often does this class appear in training data
        model['log_prior'][c] = math.log(n_c / n_total)
 
        # for each feature, compute mean and std across all class-c emails
        means_c = []
        stds_c  = []
 
        for f in range(n_features):
            col = [rows_c[r][f] for r in range(n_c)]
            mean, std = compute_mean_std(col)
 
            # if std is 0 (all emails had the same value for this feature)
            # we use a tiny number instead to avoid dividing by zero later
            std = max(std, 0.0001)
 
            means_c.append(mean)
            stds_c.append(std)
 
        model['means'][c] = means_c
        model['stds'][c]  = stds_c
 
    return model
 
 
# compute log of the gaussian probability density
# this tells us how likely a value x is given a normal distribution with mean mu and std sigma
# using log because we'll be summing these up instead of multiplying tiny numbers
def log_gaussian(x, mu, sigma):
    # log N(x; mu, sigma) = -0.5*log(2*pi) - log(sigma) - (x-mu)^2 / (2*sigma^2)
    return (
        -0.5 * math.log(2.0 * math.pi)
        - math.log(sigma)
        - 0.5 * ((x - mu) / sigma) ** 2
    )
 
 
# classify each email in the test set
# for each email we compute a score for spam and not-spam and pick the higher one
# score = log P(class) + sum of log P(feature_i | class) for all 57 features
def predict(model, X_test):
    predictions = []
 
    for x in X_test:
        best_class = None
        best_score = float('-inf')
 
        for c in model['classes']:
            # start with the log prior for this class
            score = model['log_prior'][c]
 
            # add the log likelihood for each feature
            for f in range(len(x)):
                score += log_gaussian(x[f], model['means'][c][f], model['stds'][c][f])
 
            if score > best_score:
                best_score = score
                best_class = c
 
        predictions.append(best_class)
 
    return predictions
 
 
# compute accuracy, precision, recall, and confusion matrix
# treating spam (1) as the positive class
# tp = correctly predicted spam
# tn = correctly predicted not-spam
# fp = predicted spam but was actually not-spam (false alarm)
# fn = predicted not-spam but was actually spam (missed it)
def evaluate(y_true, y_pred):
    tp = sum(1 for a, p in zip(y_true, y_pred) if a == 1 and p == 1)
    tn = sum(1 for a, p in zip(y_true, y_pred) if a == 0 and p == 0)
    fp = sum(1 for a, p in zip(y_true, y_pred) if a == 0 and p == 1)
    fn = sum(1 for a, p in zip(y_true, y_pred) if a == 1 and p == 0)
 
    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
 
    confusion_matrix = [[tn, fp],
                        [fn, tp]]
 
    return accuracy, precision, recall, confusion_matrix
 
 
def print_confusion_matrix(matrix):
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    w = 20
    print()
    print(" " * 22 + f"{'Predicted NOT-spam':^{w}}" + f"{'Predicted SPAM':^{w}}")
    print("-" * (22 + 2 * w))
    print(f"{'Actual NOT-spam':<22}{tn:^{w}}{fp:^{w}}")
    print(f"{'Actual SPAM':<22}{fn:^{w}}{tp:^{w}}")
    print()
 
 
def main():
    filename = "spambase.data"
 
    # load data
    print(f"Loading {filename}...")
    X, y = load_data(filename)
    print(f"  {len(X)} emails, {len(X[0])} features, spam rate = {sum(y)/len(y):.3f}")
 
    # step 1 - split
    print("\nStep 1: splitting into train/test sets")
    X_train, y_train, X_test, y_test = stratified_split(X, y, test_fraction=0.5, seed=42)
    print(f"  training: {len(y_train)} emails (spam rate: {sum(y_train)/len(y_train):.3f})")
    print(f"  test:     {len(y_test)} emails (spam rate: {sum(y_test)/len(y_test):.3f})")
 
    # step 2 - train
    print("\nStep 2: training the model")
    model = train_naive_bayes(X_train, y_train)
    for c in model['classes']:
        label = "spam" if c == 1 else "not-spam"
        print(f"  P({label}) = {math.exp(model['log_prior'][c]):.4f}")
 
    # step 3 - predict
    print(f"\nStep 3: classifying {len(X_test)} test emails")
    y_pred = predict(model, X_test)
 
    # step 4 - evaluate
    accuracy, precision, recall, cm = evaluate(y_test, y_pred)
    print(f"\nResults:")
    print(f"  Accuracy  = {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision = {precision:.4f}")
    print(f"  Recall    = {recall:.4f}")
    print("\nConfusion matrix:")
    print_confusion_matrix(cm)
 
 
 
if __name__ == "__main__":
    main()