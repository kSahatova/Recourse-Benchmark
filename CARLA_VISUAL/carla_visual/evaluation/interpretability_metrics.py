import torch 

def calculate_IM1(explanation_img, ae_target_class, ae_pred_class, epsilon=1e-5):
	"""
	return: IM1 metric
	"""
	
	l2_square = torch.nn.MSELoss(reduce=None, reduction='sum')
	explanation_img = explanation_img.unsqueeze(0)

	with torch.no_grad():
		ae_target_class.eval() , ae_pred_class.eval()
		t_recon = ae_target_class(explanation_img)
		i_recon = ae_pred_class(explanation_img)

	t_error = l2_square(t_recon.flatten(), explanation_img.flatten()).item()	
	i_error = l2_square(i_recon.flatten(), explanation_img.flatten()).item()  

	im1 = round(t_error/(i_error + epsilon), 4)

	return im1


def calculate_IM2(explanation_img, ae_target_class, ae_full):
	"""
	return: IM2 metric
	"""

	explanation_img = explanation_img.unsqueeze(0)
	all_recon = ae_full(explanation_img).flatten().detach().numpy()
	e_recon = ae_target_class(explanation_img).flatten().detach().numpy()
	
	x_l1_norm = float(sum(abs(explanation_img.flatten())))
	
	return sum((e_recon - all_recon)**2) / x_l1_norm


