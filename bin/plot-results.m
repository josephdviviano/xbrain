% imitate
sig_rois = [41, 67, 82, 84, 85, 96, 97, 104, 107, 125]
titles = {'pre-SMA', 'precentral-gyrus', 'temporal', 'fusiform', 'precuneus', 'post-cingulate', 'parietal', 'IPL', 'IPL', 'TPJ'};
im_results = mean_im(sig_rois, :);
bargraph(im_results, titles)

% observe
sig_rois = [20, 117, 134];
titles = {'sup-frontal', 'angular-gyrus', 'IPS'};
ob_results = mean_ob(sig_rois, :);
bargraph(ob_results, titles)