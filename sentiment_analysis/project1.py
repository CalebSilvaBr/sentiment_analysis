from string import punctuation, digits
import numpy as np
import random


#==============================================================================
#===  PART I  =================================================================
#==============================================================================


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    ntheta = np.array(theta)
    nfeature = np.array(feature_vector)
    a = 0

    for i in range(len(ntheta)):
        a = a + (ntheta[i] * nfeature[i])

    z = label * (a + theta_0)

    if z >= 1:
        return 0
    else:
        return 1 - z


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    a = 0
    for i in range(len(feature_matrix)):
        a = a + hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)

    return a / len(feature_matrix)

    # Your code here
#    raise NotImplementedError


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    theta, theta_0 = current_theta, current_theta_0

    check = label * (np.dot(current_theta, feature_vector) + current_theta_0)

    if check <= 0:
        theta = current_theta + (label * feature_vector)
        theta_0 = current_theta_0 + label

    return theta, theta_0

def perceptron(feature_matrix, labels, T):

    size = feature_matrix.shape[1]
    theta = np.zeros((size,)) # dtype=np.float32
    theta_0 = 0.0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            (theta, theta_0) = perceptron_single_step_update(
                feature_matrix[i, :],
                labels[i],
                theta,
                theta_0)
            pass

    return theta, theta_0

def average_perceptron(feature_matrix, labels, T):

    allThetas, allThetas_0 = [], []
    theta, theta_0 = np.zeros((feature_matrix.shape[1],)), 0.0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i, :], labels[i], theta, theta_0)
            allThetas.append(theta)
            allThetas_0.append(theta_0)

    sumThetas = 0.0
    sumThetas_0 = 0.0
    for i in range(len(allThetas)):
        sumThetas = sumThetas + allThetas[i]
        sumThetas_0 = sumThetas_0 + allThetas_0[i]

    d = T*len(feature_matrix)
    return sumThetas/d, sumThetas_0/d


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):

    if label * ( np.dot(current_theta, feature_vector) + current_theta_0) <= 1:
        current_theta = np.dot((1 - eta * L), current_theta) + eta * label * feature_vector
        current_theta_0 = current_theta_0 + eta * label
    else:
        current_theta = np.dot((1 - eta * L), current_theta)

    return current_theta, current_theta_0

def pegasos(feature_matrix, labels, T, L):
    n = feature_matrix.shape[0]
    m = feature_matrix.shape[1]
    # (nsamples, nfeatures) = feature_matrix.shape
    theta = np.zeros((m,))
    theta_0 = 0.0
    t_all = [i for i in range(1, n * T + 1)]
    t_idx = 0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            eta = 1 / np.sqrt(t_all[t_idx])
            (theta, theta_0) = pegasos_single_step_update(
                feature_matrix[i, :],
                labels[i],
                L,
                eta,
                theta,
                theta_0)
            t_idx += 1

    return theta, theta_0



#==============================================================================
#===  PART II  ================================================================
#==============================================================================


##  #pragma: coderesponse template
##  def decision_function(feature_vector, theta, theta_0):
##      return np.dot(theta, feature_vector) + theta_0
##  def classify_vector(feature_vector, theta, theta_0):
##      return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1
##  #pragma: coderesponse end


def classify(feature_matrix, theta, theta_0):

    results = np.zeros(len(feature_matrix))
    for i in range(len(feature_matrix)):
        if np.dot(feature_matrix[i], theta) + theta_0 <= 0:
            results[i] = -1
        else:
            results[i] = 1

    return results

    # Your code here
#    raise NotImplementedError


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):

    train_theta, train_theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
#    val_theta, val_theta_0 = classifier(val_feature_matrix, val_labels, **kwargs)

    trained_y = classify(train_feature_matrix, train_theta, train_theta_0)
    trained_acc = accuracy(trained_y, train_labels)

    val_y = classify(val_feature_matrix, train_theta, train_theta_0)
    val_acc = accuracy(val_y, val_labels)

    return trained_acc, val_acc
    # Your code here
#    raise NotImplementedError


def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    # Your code here
#    raise NotImplementedError

    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()


def bag_of_words(texts, remove_stopword=True):
    """
    NOTE: feel free to change this code as guided by Section 3 (e.g. remove
    stopwords, add bigrams etc.)

    Args:
        `texts` - a list of natural language strings.
    Returns:
        a dictionary that maps each word appearing in `texts` to a unique
        integer `index`.
    """
    # Your code here
#    raise NotImplementedError

    indices_by_word = {}  # maps word to unique index

    if remove_stopword == False:
        stopword = []
    else:
        with open('stopwords.txt', 'r') as f:
            stopword = f.read().split('\n')

    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word: continue
            if word in stopword: continue
            indices_by_word[word] = len(indices_by_word)

    return indices_by_word


def extract_bow_feature_vectors(reviews, indices_by_word, binarize=False):
    """
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """
    # Your code here
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            feature_matrix[i, indices_by_word[word]] += 1
    if binarize:
        feature_matrix = np.array(
            [[1 if c > 0 else 0 for c in feature_matrix[i, :]] for i in range(feature_matrix.shape[0])])

        # Your code here
#        raise NotImplementedError
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()
