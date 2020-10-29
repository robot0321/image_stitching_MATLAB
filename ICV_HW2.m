clear; close all; clc;
run('lib/vlfeat-0.9.21/toolbox/vl_setup.m')

% data loading
img = {};
for i=1:6
    tmpimg = imread('data/'+string(i)+'.jpg');
    tmpimg = imresize(tmpimg(:,641:end-640,:), 0.2);
    tmpimg = imrotate(tmpimg, -90);
    img{i} = single(rgb2gray(tmpimg));
end

% getting SIFT features
[fa, da] = vl_sift(img{1});
[fb, db] = vl_sift(img{2});

% plotting SIFT feature in one image
figure(1); imshow(img{1}/255); hold on;
perm = randperm(size(fa,2)) ;
sel = perm(1:20) ;
hkey = vl_plotframe(fa(:,sel)) ;
hdes = vl_plotsiftdescriptor(da(:,sel),fa(:,sel)) ;
set(hkey,'color','y','linewidth',2) ;
set(hdes,'color','g') ;

% plotting matched sift features
[matches, scores] = vl_ubcmatch(da, db);
matchedPoint1 = [fa(1:2, matches(1,:))', ones(size(matches,2),1)];
matchedPoint2 = [fb(1:2, matches(2,:))', ones(size(matches,2),1)];
figure(2); showMatchedFeatures(img{1}, img{2}, matchedPoint1(:,1:2), matchedPoint2(:,1:2), 'montage')

%% DLT, RANSAC

numIter = 50000;
count = 0;
while(count<=numIter)
    [candidH, minh, value] = DLT(4, size(matches,2), matchedPoint1, matchedPoint2);
    diag(matchedPoint2*candidH*matchedPoint1')

    
    numIter = numIter/10;
    count = count + 1;
end



%%
for i=1:6
    figure(1);
    imshow(img{i});
    pause(0.5);
end

