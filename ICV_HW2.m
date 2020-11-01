clear; close all; clc;
run('lib/vlfeat-0.9.21/toolbox/vl_setup.m')

% data loading
img = {};
for i=1:5
    tmpimg = imread('data3/'+string(i)+'.jpg');
    tmpimg = imresize(tmpimg(:,641:end-640,:), 0.2);
    tmpimg = imrotate(tmpimg, -90);
    img{i} = single(rgb2gray(tmpimg));
end

Hsave = {};
MatchIdxsave = {};
mpts1save = {};
mpts2save = {};
for imgidxk=1:4
    % getting SIFT features
    [fa, da] = vl_sift(img{imgidxk});
    [fb, db] = vl_sift(img{imgidxk+1});

    if imgidxk==1
        % plotting SIFT feature in one image
        figure(1); imshow(img{1}/255); hold on;
        perm = randperm(size(fa,2)) ;
        sel = perm(1:20) ;
        hkey = vl_plotframe(fa(:,sel)) ;
        hdes = vl_plotsiftdescriptor(da(:,sel),fa(:,sel)) ;
        set(hkey,'color','y','linewidth',2) ;
        set(hdes,'color','g') ;
    end

    % plotting matched sift features
    [matches, scores] = vl_ubcmatch(da, db);
    matchedPoint1 = [fa(1:2, matches(1,:))', ones(size(matches,2),1)];
    matchedPoint2 = [fb(1:2, matches(2,:))', ones(size(matches,2),1)];
    if imgidxk==1
        figure(2); showMatchedFeatures(img{1}, img{2}, matchedPoint1(:,1:2), matchedPoint2(:,1:2), 'montage')
    end

    %% DLT, RANSAC

    numIter = 50000;
    IterHistory = [];
    count = 0;
    criterion = 0.3;
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

    if imgidxk==1
        figure(3); showMatchedFeatures(img{1}, img{2}, matchedPoint1(maxMatchedIdx,1:2), matchedPoint2(maxMatchedIdx,1:2), 'montage')
        figure(4); plot(log(IterHistory)); ylabel('N (log-scale)'); xlabel('iteration');
    end
    
    Hsave{imgidxk} = maxH;
    MatchIdxsave{imgidxk} = maxMatchedIdx;
    mpts1save{imgidxk} = matchedPoint1;
    mpts2save{imgidxk} = matchedPoint2;
end

%% show total result
figure(5);

for imgidxk=1:4
    tform = fitgeotrans(mpts1save{imgidxk}(MatchIdxsave{imgidxk},1:2), mpts2save{imgidxk}(MatchIdxsave{imgidxk},1:2), 'projective');
    Jregistered1 = imwarp(img{imgidxk}/255,tform,'OutputView',imref2d(size(img{imgidxk+1})));
    subplot(1,5,imgidxk);
    imshowpair(img{imgidxk+1}, Jregistered1)
end

%% show matched point in a pair
pairidx = 3;
figure(6); showMatchedFeatures(img{pairidx}, img{pairidx+1}, mpts1save{pairidx}(MatchIdxsave{pairidx},1:2), mpts2save{pairidx}(MatchIdxsave{pairidx},1:2), 'montage')

%% stitched image
tform = {};
tform{1} = projective2d(Hsave{1}'*Hsave{2}');
tform{2} = projective2d(Hsave{2}');
tfrom{3} = projective2d(eye(3));
tform{4} = projective2d(Hsave{3}');
tform{5} = projective2d(Hsave{3}'*Hsave{4}');

A = zeros(256*256, 2);
for i=1:256
    for j=1:256
        A(256*(i-1)+j,1) = i;
        A(256*(i-1)+j,2) = j;
    end
end

newCord = {};
for i=[1,2]
    newCord{i} = tform{i}.transformPointsForward(A);
end
newCord{3} = A;
for i=[4,5]
    newCord{i} = tform{i}.transformPointsInverse(A);
end

%%   
figure(8);
land = zeros(500,1000);
for imidx = [3,2,1,4,5]
    for i=1:size(A,1)
        land(round(173+newCord{imidx}(i,2)), round(473+newCord{imidx}(i,1))) = img{imidx}(A(i,2), A(i,1))/255;
    end
end
imshow(land)

%% ###############################################
newCord = {};
for i=[1,2]
    newCord{i} = tform{i}.transformPointsInverse(A);
end
newCord{3} = A;
for i=[4,5]
    newCord{i} = tform{i}.transformPointsForward(A);
end

figure(9);
land = zeros(500,1000);


for imidx = [2]
    for i=1:500
        for j=1:1000
            coord = tform{imidx}.transformPointsInverse([i,j]);
            if 1<=coord(1) && coord(1)<=256 && 1<=coord(2) && coord(2)<=256
                land(i,j) = img{2}(round(coord(1)), round(coord(2)))/255;
            end
        end
    end
end
imshow(land)


%%
AA = zeros(500*1000, 2);
for i=1:500
    for j=1:1000
        AA(1000*(i-1)+j,1) = j-473;
        AA(1000*(i-1)+j,2) = i-173;
    end
end
tform_test = projective2d(Hsave{2}');
B = tform_test.transformPointsForward(AA);

figure(21);
scatter(B(:,1), B(:,2)); hold on;
scatter(A(:,1), A(:,2));
