#%%
import csv

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from deap import algorithms, base, creator, gp, tools
from sklearn.model_selection import KFold, train_test_split
import operator
import functools
import copy
import os

#%%
# Parameters
NUM_AGES = 3
GP_POP_SIZE = 300
GP_GENERATIONS = 50
NN_EPOCHS = 1000
CV_FOLDS = 3
MIXUP_ALPHA = 0.6
BATCH_SIZE = 20

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


#%%
###### Set your data and output direction######
folder_path = "/Path/to/your/data/"
output_path = "/Path/to/your/output/output.csv"

test_age = int(input("Input your age (40-69):"))

csv_files = []
for f in os.listdir(folder_path):
    if f.endswith('.csv'):
        csv_files.append(f)

result_list = []

for csv_file in csv_files:
    try:
        file_path = os.path.join(folder_path, csv_file)
        file_name = os.path.splitext(csv_file)[0]

        data = pd.read_csv(file_path)
        gene = data.columns[2]
        miu = data.loc[1,'re']
        theta = data.loc[1,'re']

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

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

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

        age_bins = np.digitize(ages, bins=[50,60])  


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

                aug_combined = np.column_stack([aug_ages, aug_expr])
                aug_train_val = np.array(
                    [x for x in aug_combined if not any(np.array_equal(x, y) for y in final_test_points)])

                X_train, X_val, y_train, y_val = train_test_split(
                    aug_train_val[:, 0],
                    aug_train_val[:, 1],
                    test_size=0.2,
                    stratify=np.digitize(aug_train_val[:, 0], bins=[50, 60])  
                )
                #print(X_train)
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

                # 选择最佳GP公式
                best_gp = min(hof, key=lambda ind: gp_fitness(ind)[0])
                gp_func = toolbox.compile(expr=best_gp)
                print(f"Best GP formula: {str(best_gp)[:70]}...")

                # 准备神经网络数据
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train).unsqueeze(1),
                    torch.FloatTensor([gp_func(a) for a in X_train]).unsqueeze(1),
                    torch.FloatTensor(y_train).unsqueeze(1))

                # 模型训练
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

                    model.eval()
                    with torch.no_grad():
                        val_gp = torch.FloatTensor([gp_func(a) for a in X_val]).unsqueeze(1)
                        val_pred = model(torch.FloatTensor(X_val).unsqueeze(1), val_gp)
                        val_loss = criterion(val_pred, torch.FloatTensor(y_val).unsqueeze(1))
                        val_loss_history.append(val_loss.item())

                    scheduler.step(val_loss)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        early_stop_counter = 0
                        best_weights = copy.deepcopy(model.state_dict())
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= 10:
                            break

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
        fold_results, all_preds = enhanced_cross_validation(ages, expr_values)

        best_fold = np.argmin([r['test_loss']+0.1*r['gp_complexity'] for r in fold_results])
        best_model = EnhancedHybridNN()
        best_model.load_state_dict(fold_results[best_fold]['model_state'])

        gp_func = toolbox.compile(expr=fold_results[best_fold]['gp_formula'])


        def predict(age):
            best_model.eval()

            age_tensor = torch.FloatTensor([age]).unsqueeze(0)
            gp_feature = torch.FloatTensor([gp_func(age)]).unsqueeze(0)

            with torch.no_grad():
                return best_model(age_tensor, gp_feature).item()

        prediction = predict(test_age)* theta + miu

        result_list.append([file_name, prediction])

        print(f"Processed {file_name}: Prediction = {prediction:.4f}")

    except Exception as e:
        print(f"Error processing file {csv_file}: {str(e)}")
        result_list.append([file_name, "ERROR"])

with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Gene', 'Expression Level'])  
        writer.writerows(result_list)  

print("\nProcessing completed. Results saved to:", output_path)
