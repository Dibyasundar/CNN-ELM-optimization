function[]=ELM_MFO_prog()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is an implementation of CNN-ELM algorithm where the Moth-flame optimization is used to update the weights.
% I have used MNIST data set as an example for 5000 samples.
% This is a complete CPU implementation 
% The code is tested to work on a Ubuntu machine with i7-4770 CPU and 8 GB RAM and MATLAB 2018a
% The moth-flame code is adapted and modified from the code uploaded by Dr. Seyedali Mirjalili
% The user of this code must cite his paper Mirjalili, S., 2015. Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm. Knowledge-Based Systems, 89, pp.228-249.
% 
%
%
% Author: Dibyasundar Das, Ph.D. Scholar, Dept. of CSE, NIT, Rourkela
% Mail id: dibyaresearch@gmail.com, dibyasundar@ieee.org
% Github link: https://github.com/Dibyasundar
% Last Date modified: 31st Jan 2019 at 04:20 pm  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clear;clc;warning off;close all;
    if ~exist('MNIST_data.mat','file')
        get_mnist_data()
    end
    t=load('MNIST_data.mat');
    data=t.train_img(:,:,:,1:5000);
    cls=t.train_label(1:5000,:)+1;
    [train_data,test_data,valid_data,train_cls,test_cls,valid_cls]=...
        devide_dataset(data,cls,0.8,0.75,'random','shuffle');
    clear t;
    %%%% set model parameter
    parma.num_hiden=5;
    parma.sz_hidden={[5,5,10],[5,5,15],[3,3,20],[50],[100]};
    parma.type={'conv','conv','conv','fc','fc'};
    parma.input_sample=train_data(:,:,:,1);
    parma.weigt_init={'rand(-1,1)','rand(-1,1)','rand(-1,1)','ortho','ortho'};
    parma.pool={[3,3],[2,2],[2,2],[],[]};
    parma.acti={'Relu','Relu','Relu','Linear','Linear'};
    parma.sz_output=size(unique(train_cls),1);
    parma.bias=0;
    %%%%% Training model
    model=train_model_opti(train_data,train_cls,valid_data,valid_cls,parma);
    
    %%%% Testing model
    [acc,output,conf]=test_model(test_data,test_cls,model)
end
function [Best_flame_pos]=train_model_opti(train_data,train_cls,valid_data,valid_cls,parma)
    display('MFO is optimizing your problem');

    %Initialize the positions of moths
    N=10;
    Max_iteration=100;
    parfor i=1:N
        Moth_pos(i)=train_model(train_data,train_cls,set_model(parma));
    end

    Convergence_curve=zeros(1,Max_iteration);

    Iteration=1;

    % Main loop
    while Iteration<Max_iteration+1

        % Number of flames Eq. (3.14) in the paper
        Flame_no=round(N-Iteration*((N-1)/Max_iteration));

        parfor i=1:size(Moth_pos,2)
            [~,~,~,Moth_fitness(1,i)]=test_model(valid_data,valid_cls,Moth_pos(i));  
        end

        if Iteration==1
            % Sort the first population of moths
            [fitness_sorted I]=sort(Moth_fitness);
            sorted_population=Moth_pos(I);

            % Update the flames
            best_flames=sorted_population;
            best_flame_fitness=fitness_sorted;
        else

            % Sort the moths
            double_population=[previous_population,best_flames];
            double_fitness=[previous_fitness best_flame_fitness];

            [double_fitness_sorted I]=sort(double_fitness);
            double_sorted_population=double_population(I);

            fitness_sorted=double_fitness_sorted(1:N);
            sorted_population=double_sorted_population(1:N);

            % Update the flames
            best_flames=sorted_population;
            best_flame_fitness=fitness_sorted;
        end

        % Update the position best flame obtained so far
        Best_flame_score=fitness_sorted(1);
        Best_flame_pos=sorted_population(1);

        previous_population=Moth_pos;
        previous_fitness=Moth_fitness;

        % a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
        a=-1+Iteration*((-1)/Max_iteration);

        for i=1:size(Moth_pos,2)
            b=0.2;
            if i<=Flame_no % Update the position of the moth with respect to its corresponsing flame
                distance_to_flame=cell_abs(cell_sub(sorted_population(i).weight,Moth_pos(i).weight));
                t=vec_mul(vec_sum(cell_rand(distance_to_flame),1),(a-1));
                Moth_pos(i).weight=cell_sum(cell_mul(cell_mul(distance_to_flame,cell_exp(vec_mul(t,b))),cell_cos(vec_mul(t,2*pi))),sorted_population(i).weight);
            elseif i>Flame_no % Upaate the position of the moth with respct to one flame
                pp=randperm(Flame_no,1);
                distance_to_flame=cell_abs(cell_sub(sorted_population(pp).weight,Moth_pos(i).weight));
                t=vec_mul(vec_sum(cell_rand(distance_to_flame),1),(a-1));
                Moth_pos(i).weight=cell_sum(cell_mul(cell_mul(distance_to_flame,cell_exp(vec_mul(t,b))),cell_cos(vec_mul(t,2*pi))),sorted_population(pp).weight);
            end

        end

        Convergence_curve(Iteration)=Best_flame_score;

        % Display the iteration and best optimum obtained so far
        if mod(Iteration,2)==0
            display(['At iteration ', num2str(Iteration), ' the best fitness is ', num2str(Best_flame_score), ' and mean is ', num2str(mean(Moth_fitness))]);
        end
        Iteration=Iteration+1; 
    end
    Best_flame_pos=train_model(train_data,train_cls,Best_flame_pos);
    save('Model.mat','Best_flame_pos');
end
function[a]=cell_abs(a)
    for i=1:size(a,2)
        a{i}=abs(a{i});
    end
end
function[a]=cell_sum(a,b)
    for i=1:size(a,2)
        a{i}=a{i}+b{i};
    end
end
function[a]=cell_mul(a,b)
    for i=1:size(a,2)
        a{i}=a{i}.*b{i};
    end
end
function[a]=cell_sub(a,b)
    for i=1:size(a,2)
        a{i}=a{i}-b{i};
    end
end
function[b]=cell_rand(a)
    for i=1:size(a,2)
        b{i}=weight_init(size(a{i},1),size(a{i},2),1,1,'rand(-1,1)');
    end
end
function[b]=cell_exp(a)
    for i=1:size(a,2)
        b{i}=exp(a{i});
    end
end
function[b]=cell_cos(a)
    for i=1:size(a,2)
        b{i}=cos(a{i});
    end
end
function[a]=vec_sum(a,b)
    for i=1:size(a,2)
        a{i}=a{i}+b;
    end
end
function[a]=vec_mul(a,b)
    for i=1:size(a,2)
        a{i}=a{i}*b;
    end
end
function[acc,output,conf,error]= test_model(data,cls,model)

    for j=1:size(data,4)
        d=data(:,:,:,j);first_fc=1;
        for i=1:size(model.weight,2)
            w=model.weight{i};
            if strcmp(model.type(i),'conv')
                temp=[];
                sz_pool=model.pool{i};
                for k=1:size(w,4)
                    temp1=convn(d,w(:,:,:,k),'valid');
                    temp1=img_pooling(temp1,sz_pool(1),sz_pool(1),sz_pool(2),'mean');
                    temp=cat(3,temp,temp1);
                end
                d=temp;
            elseif strcmp(model.type(i),'fc')
                if first_fc==1
                    d=(d(:))';
                    first_fc=0;
                end
                if size(model.weight{i},1)==length(d)+1
                    d(end+1)=1;
                end
                d=d*model.weight{i};
            end
            d=activation_fun(d,model.acti{i});
        end
        out(j,:)=d;
    end
    [~,output]=max(out');
    conf=confusionmat(output',cls);
    acc=sum(diag(conf))/sum(sum(conf));
    error=mean2((full(ind2vec(cls'))'-out).^2);
end
function[model]= train_model(data,cls,model)
    
    for j=1:size(data,4)
        d=data(:,:,:,j);first_fc=1;
        for i=1:size(model.weight,2)-1
            w=model.weight{i};
            if strcmp(model.type(i),'conv')
                temp=[];
                sz_pool=model.pool{i};
                for k=1:size(w,4)
                    temp1=convn(d,w(:,:,:,k),'valid');
                    temp1=img_pooling(temp1,sz_pool(1),sz_pool(1),sz_pool(2),'mean');
                    temp=cat(3,temp,temp1);
                end
                d=temp;
            elseif strcmp(model.type(i),'fc')
                
                if first_fc==1
                    d=(d(:))';
                    first_fc=0;
                end
                if size(model.weight{i},1)==length(d)+1
                    d(end+1)=1;
                end
                d=d*model.weight{i};
            end
            d=activation_fun(d,model.acti{i});
        end
        out(j,:)=d;
    end
    if size(model.weight{i},1)==size(data,2)+1
        data(:,end+1)=1;
    end
    model.weight{end}=pinv(out)*full(ind2vec(cls'))';
end
function [img_n]=img_pooling(img,m,n,k,type)
    i=1;
    j=1;
    p=1;
    q=1;
    [s,t]=size(img);
    while i<s
        while j<t
            if strcmp(type,'max')
                img_n(p,q)=max(max(img(i:min(i+m,s),j:min(j+n,t))));
            elseif strcmp(type,'min')
                img_n(p,q)=min(min(img(i:min(i+m,s),j:min(j+n,t))));
            elseif strcmp(type,'mean')
                img_n(p,q)=mean(mean(img(i:min(i+m,s),j:min(j+n,t))));
            end
            q=q+1;
            j=j+k;
        end
        p=p+1;
        q=1;
        j=1;
        i=i+k;
    end
end
function [model]=set_model(parma)
    k=1;input_sample=parma.input_sample;first_fc=1;
    for i=1:parma.num_hiden
        if strcmp(parma.type{i},'conv')
            m=parma.sz_hidden{i}(1);
            n=parma.sz_hidden{i}(2);
            q=size(input_sample,3);
            r=parma.sz_hidden{i}(3);
            w{k}=weight_init(m,n,q,r,parma.weigt_init{i});
            temp=[];
            sz_pool=parma.pool{i};
            for kk=1:r
                temp1=convn(input_sample,w{k}(:,:,:,kk),'valid');
                temp1=img_pooling(temp1,sz_pool(1),sz_pool(1),sz_pool(2),'mean');
                temp=cat(3,temp,temp1);
            end
            input_sample=temp;
            k=k+1;
        else
            if first_fc==1
                m=length(input_sample(:));
                first_fc=0;
            end
            n=parma.sz_hidden{i};
            if parma.bias==1
                m=m+1;
            end
            w{k}=weight_init(m,n,1,1,parma.weigt_init{i});
            m=n;
            k=k+1;
        end
    end
    if parma.bias==1
        m=m+1;
    end
    n=parma.sz_output;
    w{k}=weight_init(m,n,1,1,'rand(0,1)');
    model.weight=w;
    model.acti=[parma.acti,{'SoftMax'}];
    model.type=[parma.type,{'fc'}];
    model.pool=[parma.pool,{[]}];
end
function[w2]=weight_init(m,n,q,r,p)
    w2=[];
    for ii=1:r
        w1=[];
        for i=1:q
            switch p
                case 'rand(0,1)'
                    w=rand(m,n);
                case 'rand(-1,1)'
                    w=-1+(rand(m,n)*2);
                case 'relu'
                    w=normrnd(0,(2/(n)),m,n);
                case 'ortho'
                    w=rand(m,n);
                    [s,d]=qr(w);
                    if m>=n
                        w=s(1:m,1:n);
                    else
                        w=d(1:m,1:n);
                    end
                otherwise
                    disp(strcat('Unkown input .....',{' '},p,' is not a valid input.... recheck input'));return;
            end
            w1=cat(3,w1,w);
        end
        w2=cat(4,w2,w1);
    end
end
function[x]=activation_fun(x,type)
    switch type
        case 'Linear'
            x=x;
        case 'Relu'
            x(x<0)=0;
        case 'Sigmoid'
            x=1./(1+exp(-x));
        case 'Tanh'
            x=tanh(x);
        case 'Softsign'
            x=(x)./(1+abs(x));
        case 'Softplus'
            x=log(1+exp(x));
        case 'Sin'
            x=sin(x);
        case 'Cos'
            x=cos(x);
        case 'Sinc'
            x(x==0)=1;
            x(x~=0)=sin(x(x~=0))./x(x~=0);
        case 'LeakyRelu'
            x(x<0)=0.001.*(x(x<0));
        case 'Logistic'
            x=1./(1+exp(-x));
        case 'Gaussian'
            x=exp(-(x.^2));
        case 'BentIde'
            x=((sqrt((x.^2)+1)-1)./2)+x;
        case 'ArcTan'
            x=atan(x);
        case 'None'
            x=x;
        case 'SoftMax'
            x=softmax(x')';
        otherwise
            disp(strcat('Unkown input .....',{' '},type,' is not a valid input.... recheck input'));return;
    end
end
function[train_data,test_data,valid_data,train_cls,test_cls,valid_cls]=devide_dataset(data,cls,tp,vp,type,sf)
    train_data=[];
    test_data=[];
    valid_data=[];
    train_cls=[];
    test_cls=[];
    valid_cls=[];
    if strcmp(type,'stratified')
        u=unique(cls);
        for i=1:size(u,1)
            d=data(:,:,:,cls==u(i));
            m=size(d,4);
            pos=randperm(m);
            temp=d(:,:,:,pos(1:int32(m*tp)));
            test_data=cat(4,test_data,d(:,:,:,pos(int32(m*tp)+1:m)));
            test_cls=cat(1,test_cls,repmat(u(i),[length(pos(int32(m*tp)+1:m)),1]));
            d=temp;
            m=size(d,4);
            pos=randperm(m);
            train_data=cat(4,train_data,d(:,:,:,pos(1:int32(m*vp))));
            train_cls=cat(1,train_cls,repmat(u(i),[length(pos(1:int32(m*vp))),1]));
            valid_data=cat(4,valid_data,d(:,:,:,pos(int32(m*vp)+1:m)));
            valid_cls=cat(1,valid_cls,repmat(u(i),[length(pos(int32(m*vp)+1:m)),1]));
        end
    elseif strcmp(type,'random')
        m=size(data,4);
        pos=randperm(m);
        temp=data(:,:,:,pos(1:int32(m*tp)));
        temp_cls=cls(pos(1:int32(m*tp)),:);
        test_data=data(:,:,:,pos(int32(m*tp)+1:m));
        test_cls=cls(pos(int32(m*tp)+1:m),:);
        d=temp;
        c=temp_cls;
        clear temp temp_cls;
        m=size(d,4);
        pos=randperm(m);
        train_data=d(:,:,:,pos(1:int32(m*vp)));
        train_cls=c(pos(1:int32(m*vp)),:);
        valid_data=d(:,:,:,pos(int32(m*vp)+1:m));
        valid_cls=c(pos(int32(m*vp)+1:m),:);
    end
    if strcmp(sf,'shuffle')
        pos=randperm(size(train_data,4));
        train_data=train_data(:,:,:,pos);
        train_cls=train_cls(pos,:);
        pos=randperm(size(test_data,4));
        test_data=test_data(:,:,:,pos);
        test_cls=test_cls(pos,:);
        pos=randperm(size(valid_data,4));
        valid_data=valid_data(:,:,:,pos);
        valid_cls=valid_cls(pos,:);
    end
end
function[]=get_mnist_data()
    if ~exist('MNIST_data.mat','file')
        if ~exist('train-images-idx3-ubyte','file')
            if ~exist('train-images-idx3-ubyte.gz','file')
                websave('train-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz');
            end
            gunzip('train-images-idx3-ubyte.gz');
            delete('train-images-idx3-ubyte.gz');
        end
        if ~exist('train-labels-idx1-ubyte','file')
            if ~exist('train-labels-idx1-ubyte.gz','file')
                websave('train-labels-idx1-ubyte.gz','http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz');
            end
            gunzip('train-labels-idx1-ubyte.gz');
            delete('train-labels-idx1-ubyte.gz');
        end
        if ~exist('t10k-images-idx3-ubyte','file')
            if ~exist('t10k-images-idx3-ubyte.gz','file')
                websave('t10k-images-idx3-ubyte.gz','http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz');
            end
            gunzip('t10k-images-idx3-ubyte.gz');
            delete('t10k-images-idx3-ubyte.gz');
        end
        if ~exist('t10k-labels-idx1-ubyte','file')
            if ~exist('t10k-labels-idx1-ubyte.gz','file')
                websave('t10k-labels-idx1-ubyte.gz','http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz');
            end
            gunzip('t10k-labels-idx1-ubyte.gz');
            delete('t10k-labels-idx1-ubyte.gz');
        end
        %% Read Train images:
        fid = fopen('train-images-idx3-ubyte', 'r', 'b');
        header = fread(fid, 1, 'int32');
        if header~=2051
            disp('File is corrupted  ... please remove the file and re-run the script... ');
            exit();
        end
        count=fread(fid,1,'int32');
        row_sz=fread(fid,1,'int32');
        col_sz=fread(fid,1,'int32');
        for i=1:count
            train_img(:,:,1,i)=reshape(fread(fid,row_sz*col_sz,'uint8').',[row_sz,col_sz]).';
        end
        %% Read Train Label:
        fid = fopen('train-labels-idx1-ubyte', 'r', 'b');
        header = fread(fid, 1, 'int32');
        if header~=2049
            disp('File is corrupted  ... please remove the file and re-run the script... ');
            exit();
        end
        count=fread(fid,1,'int32');
        train_label=fread(fid,count,'uint8');
        %% Read Test images:
        fid = fopen('t10k-images-idx3-ubyte', 'r', 'b');
        header = fread(fid, 1, 'int32');
        if header~=2051
            disp('File is corrupted  ... please remove the file and re-run the script... ');
            exit();
        end
        count=fread(fid,1,'int32');
        row_sz=fread(fid,1,'int32');
        col_sz=fread(fid,1,'int32');
        for i=1:count
            test_img(:,:,1,i)=reshape(fread(fid,row_sz*col_sz,'uint8').',[row_sz,col_sz]).';
        end
        %% Read Test Label:
        fid = fopen('t10k-labels-idx1-ubyte', 'r', 'b');
        header = fread(fid, 1, 'int32');
        if header~=2049
            disp('File is corrupted  ... please remove the file and re-run the script... ');
            exit();
        end
        count=fread(fid,1,'int32');
        test_label=fread(fid,count,'uint8');
        save('MNIST_data.mat','train_img','train_label','test_img','test_label');
        delete('*-ubyte')
    end
end
