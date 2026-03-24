import numpy as np
import random
import heapq
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mutual_info_score
from skimage.metrics import structural_similarity as SSIM
image = np.load('images.npy')
labels = np.load('labels.npy')
num_bands = image.shape[-1] 
print(image.shape)
first_image = image[0:128, 0:128, :]
nmi_matrix = np.zeros((num_bands, num_bands))
ssim_matrix = np.zeros((num_bands, num_bands))
dissimilarity_matrix = np.zeros((num_bands, num_bands))
def calculate_entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities)) if len(probabilities) > 1 else 0
for i in range(num_bands):
    for j in range(i + 1, num_bands):
        band_i = first_image[:, :, i].flatten()
        band_j = first_image[:, :, j].flatten()
        if len(np.unique(band_i)) > 1 and len(np.unique(band_j)) > 1:
            mi = mutual_info_score(band_i, band_j)
            h_i, h_j = calculate_entropy(band_i), calculate_entropy(band_j)
            if h_i > 0 and h_j > 0:
                dissimilarity_matrix[i, j] = h_i + h_j - 2 * mi
                dissimilarity_matrix[j, i] = dissimilarity_matrix[i, j]
                nmi_matrix[i, j] = 2 * mi / (h_i + h_j)
                nmi_matrix[j, i] = nmi_matrix[i, j]
            
            ssim_value = SSIM(
                first_image[:, :, i], first_image[:, :, j],
                data_range=first_image[:, :, i].max() - first_image[:, :, i].min()
            )
            ssim_matrix[i, j] = ssim_value
            ssim_matrix[j, i] = ssim_value
hybrid_similarity_matrix = nmi_matrix + ssim_matrix
def rank_bands(sim_matrix, dis_matrix, num_bands_to_select=50):
    avg_similarity = np.nanmean(sim_matrix, axis=1)
    dissimilarity = np.nanmin(dis_matrix, axis=1)
    scores = avg_similarity * dissimilarity
    return np.argsort(scores)[::-1][:num_bands_to_select]  
num_bands_to_select = 100
sr_nmi_vi_ranked_bands = rank_bands(nmi_matrix, dissimilarity_matrix, num_bands_to_select)

print(f"Top {num_bands_to_select} SR-NMI-VI Ranked Bands: {sr_nmi_vi_ranked_bands}")

POP_SIZE = 50   
N_GEN = 50     
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7
MIN_BANDS = 10  
MAX_BANDS = 20 
sr_nmi_vi_ranked_bands = sr_nmi_vi_ranked_bands.tolist() if isinstance(sr_nmi_vi_ranked_bands, np.ndarray) else sr_nmi_vi_ranked_bands
try:
    del creator.FitnessMax
    del creator.Individual
except AttributeError:
    pass

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def init_individual():
    num_selected_bands = random.randint(MIN_BANDS, MAX_BANDS)
    return creator.Individual(random.sample(sr_nmi_vi_ranked_bands, num_selected_bands))

toolbox = base.Toolbox()
toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def mutate_individual(individual):
    if random.random() < MUTATION_RATE:
        idx_to_replace = random.randint(0, len(individual) - 1)
        available_bands = list(set(sr_nmi_vi_ranked_bands) - set(individual))
        if available_bands:
            individual[idx_to_replace] = random.choice(available_bands)
    return (individual,)


def crossover(parent1, parent2):
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    
    child1_genes = list(set(parent1[:crossover_point] + parent2[crossover_point:]))
    child2_genes = list(set(parent2[:crossover_point] + parent1[crossover_point:]))

    # Ensure valid band count
    child1_genes = child1_genes[:MAX_BANDS] if len(child1_genes) > MAX_BANDS else child1_genes
    child2_genes = child2_genes[:MAX_BANDS] if len(child2_genes) > MAX_BANDS else child2_genes
    
    # Fill up to MIN_BANDS if needed
    while len(child1_genes) < MIN_BANDS:
        new_band = random.choice(sr_nmi_vi_ranked_bands)
        if new_band not in child1_genes:
            child1_genes.append(new_band)
    
    while len(child2_genes) < MIN_BANDS:
        new_band = random.choice(sr_nmi_vi_ranked_bands)
        if new_band not in child2_genes:
            child2_genes.append(new_band)

    return creator.Individual(child1_genes), creator.Individual(child2_genes)


def evaluate(individual):
    selected_indices = list(individual)
    X = image.reshape(-1, num_bands)[:, selected_indices]
    y = labels.ravel()

    # Use a subset of the dataset
    X = X[102000:109000, :]
    y = y[102000:109000]

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return (0,)  

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    if len(np.unique(y_train)) < 2:
        return (0,)
    
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return (acc + kappa + f1,)

toolbox.register("mate", crossover)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


def run_ga():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)  
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE, 
                        ngen=N_GEN, stats=stats, halloffame=hof, verbose=True)

    return hof[0]  

best_individual = run_ga()
print(f"Optimized Bands (GA-Wrapped SR-NMI-VI): {best_individual}")

X = image.reshape(-1, num_bands)[:, best_individual]
y = labels.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Naïve Bayes": GaussianNB(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"{name} - Accuracy: {acc:.4f}, Kappa: {kappa:.4f}, F1-score: {f1:.4f}")
