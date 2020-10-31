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
IterHistory = [];
count = 0;
criterion = 1;
maxInlierRatio = 0;
maxH = zeros(3,3);
maxMatchedIdx = [];
while(count<=numIter)    
    candidH = DLT(4, size(matches,2), matchedPoint1, matchedPoint2);
    crsval = [];
    for j=1:size(matchedPoint1, 1)
        crsval = [crsval, norm(cross(matchedPoint2(j,:), candidH*matchedPoint1(j,:)'))];
    end
    [sortval, sortidx] = sort(crsval);
    
    numinlier = sum(sortval < criterion);
    inlierRatio = numinlier/size(matches,2);

    if(inlierRatio > maxInlierRatio)
        maxInlierRatio = inlierRatio;
        maxH = candidH;
        maxMatchedIdx = sortidx(sortval < criterion);
    end
    
    numIter = log(1-0.99)/log(1-(maxInlierRatio)^4);
    count = count + 1;
    IterHistory = [IterHistory, numIter];
end

figure(3); showMatchedFeatures(img{1}, img{2}, matchedPoint1(maxMatchedIdx,1:2), matchedPoint2(maxMatchedIdx,1:2), 'montage')
figure(4); plot(log(IterHistory)); ylabel('N (log-scale)'); xlabel('iteration');

%%

figure(5);
landsc = zeros(300, 1300);
landsc(150-127:150+128, 650-127:650+128) = img{2}/255;
imshow(landsc);

J1 = imwarp(img{1}, projective2d(maxH'));
landsc(150-127:150+128, 522-127:522+128) = J1/255;


%%
figure(5);
tform = fitgeotrans(matchedPoint1(maxMatchedIdx,1:2), matchedPoint2(maxMatchedIdx,1:2), 'projective');
Jregistered1 = imwarp(img{1}/255,tform,'OutputView', imref2d(size(img{2})));
% Jregistered2 = imwarp(img{1}/255,'OutputView', imref2d(imsize(img{2})));
figure
imshowpair(img{2}, Jregistered1)




