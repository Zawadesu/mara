#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from deap import algorithms, base, creator, gp, tools
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import operator
import functools
import copy
import os

#%%
######Set your data path###### 
os.chdir("/Path/to/your/data/")

# Parameters 
NUM_AGES = 3
GP_POP_SIZE = 300
GP_GENERATIONS = 50
NN_EPOCHS = 1000
CV_FOLDS = 3
MIXUP_ALPHA = 0.6
BATCH_SIZE = 20

#%%
# Loading data
data = pd.read_csv("gene.csv") # YOUR INTERESTED GENE
gene = data.columns[2]
miu = data.loc[1,'re']
theta = data.loc[1,'re']

# Prepare data for model
group_means = data.groupby('Age')[gene].mean().reset_index()
group_means.columns = ['Age', 'Mean']
group_means.loc[group_means['Age'] == '40-49', 'Age'] = 45
group_means.loc[group_means['Age'] == '50-59', 'Age'] = 55
group_means.loc[group_means['Age'] == '60-69', 'Age'] = 65

ages = group_means['Age'].astype(np.int64).values
expr_values = group_means['Mean'].astype(np.float64).values

#%%
# Data augmentation：Mixup
def augment_data_mixup(x, y, num=208, alpha=0.6, boundary_extension=5):
    new_x, new_y = [], []
    x_min, x_max = np.min(x), np.max(x)

    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    for _ in range(num):
        i = np.random.randint(0, len(x) - 1)
        j = i + 1
        lam = np.clip(np.random.beta(alpha, alpha), 0.1, 0.9)
        mixed_x = lam * x[i] + (1 - lam) * x[j]
        mixed_y = lam * y[i] + (1 - lam) * y[j]

        if (np.random.rand() < 0.1) & (i == 0):
            mixed_x = x_min - np.random.uniform(0, boundary_extension)
            slope = (y[j] - y[i]) / (x[j] - x[i])
            mixed_y = y[i] - slope * (x[i] - mixed_x)
        elif (np.random.rand() < 0.1) & (i == 1):
            mixed_x = x_max + np.random.uniform(0, boundary_extension)
            slope = (y[j] - y[i]) / (x[j] - x[i])
            mixed_y = y[j] + slope * (mixed_x - x[j])

        new_x.append(mixed_x)
        new_y.append(mixed_y)

    return np.concatenate([x, new_x]), np.concatenate([y, new_y])

#%%
# GP configuration
def protected_div(a, b):
    return 1 if abs(b) < 1e-6 else a / b

def rand_uniform():
    return np.random.uniform(40, 69)

rand_const = functools.partial(rand_uniform)

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(np.add, 2, name="add")
pset.addPrimitive(np.subtract, 2, name="sub")
pset.addPrimitive(np.multiply, 2, name="mul")
pset.addPrimitive(protected_div, 2, name="div")
pset.addPrimitive(np.piecewise, 2, name="piecewise")
pset.addPrimitive(np.square, 1, name="square")
pset.addEphemeralConstant("rand_const", rand_const)
pset.renameArguments(ARG0='age')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Restrict the depth
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

# NN part
class EnhancedHybridNN(nn.Module):
    def __init__(self, gp_feature_dim=1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(1 + gp_feature_dim, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3))
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1))

    def forward(self, age, gp_feature):
        x = torch.cat([age, gp_feature], dim=1)
        features = self.feature_extractor(x)
        return self.regressor(features)

# Cross-validation
age_bins = np.digitize(ages, bins=[50,60])  # Group by ages

global_test_points = []

def enhanced_cross_validation(orig_ages, orig_expr, n_splits=3):
    kf = KFold(n_splits=n_splits)
    fold_results = []
    all_preds = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(orig_ages)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        test_ages = orig_ages[test_idx]
        test_expr = orig_expr[test_idx]

        aug_ages, aug_expr = augment_data_mixup(ages, expr_values)
        #print(aug_ages.shape)

        # Select a test point
        test_points = []
        n_select =1
        for low, high in [(45, 55), (55, 65)]:

            mask = (aug_ages >= low) & (aug_ages <= high)
            candidates = np.column_stack([aug_ages[mask], aug_expr[mask]])
            #print(candidates.shape)
            
            if len(candidates) >= n_select:
            
                selected_indices = np.random.choice(len(candidates), n_select, replace=False)
            else:
            
                selected_indices = np.random.choice(len(candidates), n_select, replace=True)

            selected = candidates[selected_indices]
            test_points.append(selected)

        fold_test_points = np.concatenate(test_points)
        global_test_points.append(fold_test_points)

        existing_test_points = np.column_stack([test_ages, test_expr])
        final_test_points = np.concatenate([existing_test_points, fold_test_points])
        test_ages = final_test_points[:, 0]
        test_expr = final_test_points[:, 1]

        # === Data set construction ===
        # Remove test points
        aug_combined = np.column_stack([aug_ages, aug_expr])
        aug_train_val = np.array(
            [x for x in aug_combined if not any(np.array_equal(x, y) for y in final_test_points)])

        # Split the training & validation data
        X_train, X_val, y_train, y_val = train_test_split(
            aug_train_val[:, 0],
            aug_train_val[:, 1],
            test_size=0.2,
            stratify=np.digitize(aug_train_val[:, 0], bins=[50, 60])  
        )
        #print(X_train)
        # Running GP 
        def gp_fitness(individual):
            try:
                func = toolbox.compile(expr=individual)
                pred = np.array([func(a) for a in X_train])
                return (np.mean((pred - y_train) ** 2),)
            except:
                return (np.inf,)

        toolbox.register("evaluate", gp_fitness)
        pop = toolbox.population(n=GP_POP_SIZE)
        hof = tools.HallOfFame(3)

        algorithms.eaMuPlusLambda(pop, toolbox,
                                  mu=100, lambda_=200,
                                  cxpb=0.7, mutpb=0.3,
                                  ngen=GP_GENERATIONS, halloffame=hof,
                                  stats=None, verbose=False)

        # The best GP function
        best_gp = min(hof, key=lambda ind: gp_fitness(ind)[0])
        gp_func = toolbox.compile(expr=best_gp)
        print(f"Best GP formula: {str(best_gp)[:70]}...")

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).unsqueeze(1),
            torch.FloatTensor([gp_func(a) for a in X_train]).unsqueeze(1),
            torch.FloatTensor(y_train).unsqueeze(1))

        # Train the model
        model = EnhancedHybridNN()
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.HuberLoss()

        best_loss = float('inf')
        early_stop_counter = 0
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        best_weights = copy.deepcopy(model.state_dict())
        train_loss_history = []
        val_loss_history = []

        for epoch in range(NN_EPOCHS):
            model.train()
            epoch_loss = 0
            for age, gp_feat, target in train_loader:
                optimizer.zero_grad()
                pred = model(age, gp_feat)
                loss = criterion(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            #avg_loss = epoch_loss / len(train_loader)
                train_loss_history.append(epoch_loss)

            # Evaluate the model: validation
            model.eval()
            with torch.no_grad():
                val_gp = torch.FloatTensor([gp_func(a) for a in X_val]).unsqueeze(1)
                val_pred = model(torch.FloatTensor(X_val).unsqueeze(1), val_gp)
                val_loss = criterion(val_pred, torch.FloatTensor(y_val).unsqueeze(1))
                val_loss_history.append(val_loss.item())

            # Adjust the learning rate
            scheduler.step(val_loss)

            # Early-stop
            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_counter = 0
                best_weights = copy.deepcopy(model.state_dict())
            else:
                early_stop_counter += 1
                if early_stop_counter >= 10:
                    break

        # Evaluate the model: test
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            test_gp = torch.FloatTensor([gp_func(a) for a in test_ages]).unsqueeze(1)
            test_pred = model(torch.FloatTensor(test_ages).unsqueeze(1), test_gp)
            test_loss = criterion(test_pred, torch.FloatTensor(test_expr).unsqueeze(1)).item()

        all_ages_tensor = torch.FloatTensor(ages).unsqueeze(1)
        all_gp_features = torch.FloatTensor([gp_func(a) for a in ages]).unsqueeze(1)
        with torch.no_grad():
            full_pred = model(all_ages_tensor, all_gp_features).squeeze().numpy()

        # Record the results
        fold_results.append({
            'fold': fold + 1,
            'gp_formula': str(best_gp),
            'model_state':model.state_dict(),
            'test_loss': test_loss,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'gp_complexity': best_gp.height,
            'full_predictions': full_pred,
            'predictions': test_pred.squeeze().numpy(),
            'test_ages': test_ages,
            'true_values': test_expr
        })
        all_preds.append(test_pred.squeeze().numpy())
        #print(train_loss_history)
    return fold_results, np.array(all_preds)

#%%
# Cross-validation
fold_results, all_preds = enhanced_cross_validation(ages, expr_values)


# Visualization
plt.figure(figsize=(14, 8))

#%%
# 1.Original data & Augmentation data
ages_new, expr_new = augment_data_mixup(ages, expr_values)
plt.scatter(ages, expr_values, s=50, c='red', zorder=10, label='Original data')
plt.scatter(ages_new, expr_new, s=50, c='blue', marker='^', alpha=0.5, zorder=10, label='Augment data')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Expression Level(log2(tpm+0.001))', fontsize=12)
plt.title('Data Augmentation Results', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%
print("Ages of test sets in each fold")
for fold in fold_results:
    print(f"Fold {fold['fold']}: {fold['test_ages']}")

# 2.Prediction curve
plt.figure(figsize=(12,6))
for res in fold_results:
    sorted_idx = np.argsort(ages)
    plt.plot(ages[sorted_idx], res['full_predictions'][sorted_idx],
             linestyle='--', alpha=0.7, linewidth=2,
             label=f'Fold {res["fold"]}')

plt.scatter(ages, expr_values, s=150, c='red',
           edgecolor='black', zorder=10, label='Original Data')
plt.title('Cross-Validation Predictions')
plt.xlabel('Age')
plt.ylabel('Expression Level(log2(tpm+0.001))')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

#%%
# 3.Evaluation
MSE = []
MAE = []
R2 = []
for fold in fold_results:
    y_true = fold['true_values']
    y_pred = fold['predictions']

    mse = mean_squared_error(np.array(y_true), np.array(y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

    MSE.append(mse)
    MAE.append(mae)
    R2.append(r2)
    print(f"Fold {fold['fold']}:")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")

# MAE MSE
group = ['Fold1', 'Fold2', 'Fold3']
x = np.arange(len(group))
width = 0.25

fig, ax = plt.subplots()
fs1 = ax.bar(x - width/2, MSE, width, label='MSE')
fs2 = ax.bar(x + width/2, MAE, width, label='MAE')
#fs3 = ax.bar(x + width, R2, width, label='R2')

ax.set_xlabel('Fold')
ax.set_title('Evaluation of Predictions for Each Fold')
ax.set_xticks(x)
ax.set_xticklabels(group)
ax.legend()
plt.show()

n_groups = 3  
index = np.arange(n_groups)
bar_width = 0.2 
for fold in fold_results:
    y_true = fold['true_values']
    y_pred = fold['predictions']

    residuals = y_true - y_pred

    if fold["fold"] == 1:
        fold1 = residuals
    elif fold["fold"] == 2:
        fold2 = residuals
    elif fold["fold"] == 3:
        fold3 = residuals
fig, ax = plt.subplots()
bar1 = ax.bar(index - bar_width, fold1, bar_width, label='Fold 1', color='#d9d9d9')
bar2 = ax.bar(index, fold2, bar_width, label='Fold 2', color='#b3cde0')
bar3 = ax.bar(index + bar_width, fold3, bar_width, label='Fold 3', color='#fbb4ae')

ax.set_xlabel('Test points')
ax.set_ylabel('Residuals')
ax.set_title('Prediction Residuals for Each Fold')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(['1', '2', '3'])
plt.grid(axis='y', alpha=0.3)
ax.legend()
plt.show()
'''
plt.figure(figsize=(14, 5))
residuals = [y_true] - [y_pred]
plt.bar(ages[sorted_idx], residuals, width=4, color='skyblue', edgecolor='navy')
plt.axhline(0, color='red', linestyle='--')
plt.title('Prediction Residual Analysis')
plt.xlabel('Age')
plt.ylabel('Residual')
plt.grid(axis='y', alpha=0.3)
plt.show()
'''
#%%
# 4.Lost curve

train_loss_history = [f['train_loss_history'] for f in fold_results]
#val_loss_history = [f['val_loss_history'] for f in fold_results]

min_length = min(len(hist) for hist in train_loss_history)
truncated_histories = [hist[:min_length] for hist in train_loss_history]

mean_train_loss = np.array(truncated_histories)
mean_loss = np.mean(mean_train_loss, axis=0)
std_loss = np.std(mean_train_loss, axis=0)
epochs = np.arange(1, min_length + 1)
#min_length1 = min(len(hist) for hist in val_loss_history)
#mean_val_loss = np.mean([hist[:min_length1] for hist in val_loss_history], axis=0)

plt.figure(figsize=(10, 6))  
plt.style.use("seaborn-v0_8-darkgrid")
for i, loss in enumerate(mean_train_loss):
    plt.plot(epochs, loss, color="grey", alpha=0.4, linewidth=1, label=f"Fold {i+1}")
#%%
# Average & Standard deviation
plt.plot(epochs, mean_loss, color="#FF4B4B", linewidth=2.5, label="Mean Loss")
plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color="#FF4B4B", alpha=0.1)


#plt.plot(mean_train_loss, label='Average Train Loss', color='darkblue', linewidth=2)
#plt.plot(mean_val_loss, label='Average Val Loss', color='darkorange', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Cross-Validation Training Loss Curves(Mean ± Std)')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# Saving the model
best_fold = np.argmin([r['test_loss']+0.1*r['gp_complexity'] for r in fold_results])
best_model = EnhancedHybridNN()
best_model.load_state_dict(fold_results[best_fold]['model_state'])
torch.save({
    'model_state_dict': best_model.state_dict(),
    'gp_formula': fold_results[best_fold]['gp_formula'],
    'ages_mean': np.mean(ages),
    'ages_std': np.std(ages)
}, 'best_hybrid_model.pth')



# Run the model
gp_func = toolbox.compile(expr=fold_results[best_fold]['gp_formula'])
def predict(age, model_path='best_hybrid_model.pth'):

    best_model.eval()

    age_tensor = torch.FloatTensor([age]).unsqueeze(0)
    gp_feature = torch.FloatTensor([gp_func(age)]).unsqueeze(0)

    with torch.no_grad():
        return best_model(age_tensor, gp_feature).item()


# Results
print(str(fold_results[best_fold]['gp_formula']))
test_age = int(input("Input your age (40-69): "))
print(f"\n (Age={test_age}):")
print(f"Predicted expression of the gene: {predict(test_age)*theta+miu:.2f}")
#print(f"function:{best_fold}")
