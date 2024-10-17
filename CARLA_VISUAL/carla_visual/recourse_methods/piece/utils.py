import numpy as np
import pandas as pd 
import copy

from typing import List, Tuple, Optional

import torch 
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt 

from .model import HurdleModel


def get_misclassificaiton(test_loader, classifier, k_misclf_num):
	count = 0
	for i, data in enumerate(test_loader):
		image, label = data
		label = label.detach().numpy()[0]
		prediction = torch.argmax(classifier(image)[0]).detach().numpy()
		if label != prediction:
			count += 1
			if count == k_misclf_num: 
				break
	original_query_idx = i
	original_query_img = image
	original_query_label = label
	original_query_prediction = prediction 
	
	print("Label:", int(label))
	print("Prediction:", prediction)

	return original_query_idx, original_query_img, original_query_label, original_query_prediction


def get_misclassifications(model: nn.Module,
                           data_loader: DataLoader,
                           device: str) -> List[Tuple[torch.Tensor, int, int]]:
    """
    Get misclassified samples from a PyTorch model.

    Args:
    model (nn.Module): The PyTorch model to evaluate.
    data_loader (DataLoader): The DataLoader containing the evaluation data.
    device (str): The device to run the evaluation on ('cuda' or 'cpu').
    num_samples (int): Maximum number of misclassified samples to return.

    Returns:
    List[Tuple[torch.Tensor, int, int]]: List of tuples containing (misclassified_sample, true_label, predicted_label).
    """
    model.eval()
    model = model.to(device)

    misclassified = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs[0], 1)

            # Find misclassified samples
            mask = (predicted != labels)
            misclassified.extend(list(zip(inputs[mask], labels[mask], predicted[mask])))
	
    return misclassified 


def optimize_z0(G, C, I, nz, num_iterations=10000, learning_rate=0.1, verbose=0):
    # Initialize z0 randomly
    z0 = torch.randn(1, nz, 1, 1, requires_grad=True)
    #z = z.view(z.size(0), nz, 1, 1)

    # Define optimizer
    optimizer = optim.Adam([z0], lr=learning_rate)
    
    for i in range(num_iterations):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        G_z0 = G(z0)
        C_G_z0 = C(G_z0)
        C_I = C(I)
        
        # Compute loss
        loss = torch.norm(C_G_z0 - C_I, p=2)**2 + torch.norm(G_z0 - I, p=2)**2
        
        # Backward pass
        loss.backward()
        
        # Update z0
        optimizer.step()
        
        if verbose and i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")
    
    return z0


def return_feature_contribution_data(data_loader, cnn, num_classes=10):
	"""
	This function constructs a so-called latent dataset, i.e. a 
	dictionary with the logits  as values and predictions of the classes for these logits as keys 
	"""
	pred_idx = dict() 

	for class_name in list(range(num_classes)):
		pred_idx[class_name] = list()
		
	for i, data in enumerate(data_loader):
		# print progress
		if i % 10000 == 0:
			print(  100 * round(i / len(data_loader), 2), "% complete..."  )     
		image, label = data
		label = int(label.detach().numpy())
		acts = cnn(image)[1][0].detach().numpy()
		pred = int(torch.argmax(  cnn(image)[0]  ).detach().numpy()) 
		pred_idx[pred].append(acts.tolist())
				
	return pred_idx

		
def get_data_for_feature(dist_data, target_class, feature_map_num):
	data = np.array(dist_data[target_class])
	data = data.T[feature_map_num].T.reshape(data.shape[0],1)
	return data


def get_distribution_name(dist):
	if dist.fixed_location == True:
		return dist.rv.name + " With Fixed 0 Location"
	else:
		return dist.rv.name

	
def acquire_feature_probabilities(latent_dataset, original_label, cnn, original_query_img=None, alpha=0.05):
	query_features = cnn(original_query_img)[1][0]
	digit_weights = cnn.classifier[0].weight[original_label]

	fail_results = list()
	succeed_results = list()
	high_results = list()
	low_results = list()
	expected_values = list()
	probability = list()
	p_values = list()
	distribution_type = list()

	for i in range(len(query_features)):

		data = get_data_for_feature(latent_dataset, original_label, feature_map_num=i)
		data = data.T[0].T
		feature_value = float(query_features[i])

		dist_examine = HurdleModel(data, value=feature_value, p_value=alpha)
		fail_results.append(dist_examine.bern_fail_sig())  
		succeed_results.append(dist_examine.bern_success_sig())   
		high_results.append(dist_examine.high_cont_sig())  
		low_results.append(dist_examine.low_cont_sig())  
		expected_values.append(dist_examine.get_expected_value())  
		probability.append(dist_examine.get_prob_of_value())
		p_values.append(dist_examine.test_fit())
		distribution_type.append(get_distribution_name(dist_examine))

	df = pd.DataFrame()
	df['Feature Map'] = list(range(len(query_features)))
	df['Contribution'] = query_features.detach().numpy() * digit_weights.detach().numpy()
	df['Bern Fail'] = fail_results
	df['Bern Success'] = succeed_results
	df['Cont High'] = high_results
	df['Cont Low'] = low_results
	df['Expected Value'] = expected_values
	df['Probability of Event'] = probability
	df['Distribtuion p-value KsTest'] = p_values
	df['Dist Type'] = distribution_type

	pd.set_option('display.float_format', lambda x: '%.4f' % x)
	return df


def save_query_and_gan_xp_for_final_data(I_e, cnn, z, G, z_e, original_query_image, name, rand_num):
	numpy_org_image = original_query_image.detach().numpy().reshape(28,28)
	f, axarr = plt.subplots(1,3)
	axarr[0].imshow(numpy_org_image)
	axarr[0].axis('off')
	axarr[0].title.set_text('Query')
	axarr[1].imshow(G(z).detach().numpy().reshape(28,28))
	axarr[1].axis('off')
	axarr[1].title.set_text('GAN Estimation')
	axarr[2].imshow(I_e.detach().numpy().reshape(28,28))
	axarr[2].axis('off')
	axarr[2].title.set_text('Explanation')
	plt.savefig('Explanations/' + name + "_" + str(rand_num) + '.pdf')


def modifying_exceptional_features(df, target_class, query_activations):
	"""
	Change all exceptional features to the expected value for each PDF
	return: tensor with all exceptional features turned into "expected" feature values for c'
	"""

	ideal_xp = query_activations.clone().detach()

	for idx, row in df.sort_values('Probability of Event', ascending=True).iterrows():  # from least probable feature to most probable
		feature_idx = int(row['Feature Map'])  
		expected_value = row['Expected Value'] 
		ideal_xp[feature_idx] = expected_value
	return ideal_xp


def filter_df_of_exceptional_noise(df, target_class, cnn, alpha=0.05):
	"""
	Take the DataFrame, and remove rows which are exceptional features in c' (counterfactual class) but not candidate for change.
	return: dataframe with only relevant features for PIECE algorithm

	alpha is the probability threshold for what is "excetional" or "weird" in the image.
	"""

	df_copy = copy.deepcopy(df)
	df_new = df_copy[df_copy['Probability of Event'] < alpha]

	df_new['flag'] = 0
	digit_weights = cnn.classifier[0].weight[target_class]

	for idx, row in df_new.iterrows():
		feature_idx = int(row['Feature Map'])  

		if row['Bern Fail']:  # if it's unusual to not activate, but it's negative
			if digit_weights[feature_idx] < 0: 
				df_new.at[feature_idx, 'flag'] = 1
		if row['Cont High']:  # if it's high, but positive
			if digit_weights[feature_idx] > 0: 
				df_new.at[feature_idx, 'flag'] = 1
		if row['Cont Low'] :  # if it's low, but negative
			if digit_weights[feature_idx] < 0: 
				df_new.at[feature_idx, 'flag'] = 1

	exceptional_noise_idx = df_new[df_new.flag == 0].index.tolist()
	print('Exceptions(?): ', df_new[df_new.flag == 1].index.tolist())
	df_copy = df_copy.drop(index=exceptional_noise_idx, axis=0)
	print('The length of the latent datset before and after filtering:', df.shape, '|', df_copy.shape)
	print('Number of noisy exceptional features deleted:', len(exceptional_noise_idx))

	return df_copy


def optim_PIECE(G, cnn, x_prime, z_e, n_iterations=500):
	"""
	Step 3 of the PIECE algorithm
	returns: z prime
	"""
	criterion = nn.MSELoss()
	optimizer = optim.Adam([z_e], lr=0.001)

	for i in range(n_iterations):

		optimizer.zero_grad()
		logits, x_e = cnn(G(z_e))
		loss = criterion(x_e[0], x_prime)

		loss.backward()  
		optimizer.step()  

		if i % 50 == 0:
			print("Loss:", loss.item())

	return z_e