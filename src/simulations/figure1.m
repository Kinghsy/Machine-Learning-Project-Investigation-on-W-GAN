clear all

X = -5:0.1:5;
X=X'
P1 = normpdf(X,0,1);;
P1=P1/sum(P1)

d=[-4:0.1:4];
% y = reshape(y,length(x2),length(x1));
k=1
for i= d
P2 = normpdf(X,i, 1);
% P2 = normpdf(X,0,2^i)+unifpdf(X)/100;
P2=P2/sum(P2)   
if i==2
    subplot(2,3,1)
    plot(X,P1,X,P2,'LineWidth',2)
    xlabel('x')
    xlim([-5 5])
    ylim([0 0.4])
    lg=legend('X_1\sim (0,1)','X_2\sim (d,1)')
    set(lg,'color','none');
end
KL=kldiv(X,P1,P2,'js')  
[f, fval] = emd(X,X, P1, P2, @gdf);
KLS(k)=KL;
EMS(k)=fval;
k=k+1
end
subplot(2,3,4)
plot(d,KLS,d,EMS,'LineWidth',2)
xlabel('d')
ylabel('measure')
lg=legend('JS','EMD')
set(lg,'color','none');
%

P1 = normpdf(X,0,0.3);;
P1=P1/sum(P1)


% y = reshape(y,length(x2),length(x1));
k=1
for i= d
P2 = normpdf(X,i, 0.3);
% P2 = normpdf(X,0,2^i)+unifpdf(X)/100;
P2=P2/sum(P2)   
if i==2
    subplot(2,3,2)
    plot(X,P1,X,P2,'LineWidth',2)
    xlabel('x')
    xlim([-5 5])
        ylim([0 0.4])
    lg=legend('X_1\sim (0,0.3)','X_2\sim (d,0.3)')
    set(lg,'color','none');
end
KL=kldiv(X,P1,P2,'js')  
[f, fval] = emd(X,X, P1, P2, @gdf);
KLS(k)=KL;
EMS(k)=fval;
k=k+1
end
KLS(isnan(KLS))=1;
subplot(2,3,5)
plot(d,KLS,d,EMS,'LineWidth',2)
xlabel('d')
ylabel('measure')
lg=legend('JS','EMD')
set(lg,'color','none');
%
%
P1 = normpdf(X,0,0.1);;
P1=P1/sum(P1)


% y = reshape(y,length(x2),length(x1));
k=1
for i= d
P2 = normpdf(X,i, 0.1);
% P2 = normpdf(X,0,2^i)+unifpdf(X)/100;
P2=P2/sum(P2)   
if i==2
    subplot(2,3,3)
    plot(X,P1,X,P2,'LineWidth',2)
    xlabel('x')
    xlim([-5 5])
    ylim([0 0.4])
    lg=legend('X_1\sim (0,0.1)','X_2\sim (d,0.1)')
    set(lg,'color','none');
end
KL=kldiv(X,P1,P2,'js')  
if i==0
KL=0
end
[f, fval] = emd(X,X, P1, P2, @gdf);
KLS(k)=KL;
EMS(k)=fval;
k=k+1
end
KLS(isnan(KLS))=1;
KLS()
subplot(2,3,6)
plot(d,KLS,d,EMS,'LineWidth',2);hold on;
xlabel('d')
ylim([0 4])
ylabel('measure')
lg=legend('JS','EMD')
set(lg,'color','none');

function [f] = gdm(F1, F2, Func)
[m a] = size(F1);
[n a] = size(F2);
for i = 1:m
    for j = 1:n
        f(i, j) = Func(F1(i, 1:a), F2(j, 1:a));
    end
end
f = f';
f = f(:);

end

function KL = kldiv(varValue,pVect1,pVect2,varargin)
if ~isequal(unique(varValue),sort(varValue)),
    warning('KLDIV:duplicates','X contains duplicate values. Treated as distinct values.')
end
if ~isequal(size(varValue),size(pVect1)) || ~isequal(size(varValue),size(pVect2)),
    error('All inputs must have same dimension.')
end
if (abs(sum(pVect1) - 1) > .00001) || (abs(sum(pVect2) - 1) > .00001),
    error('Probablities don''t sum to 1.')
end
if ~isempty(varargin),
    switch varargin{1},
        case 'js',
            logQvect = log2((pVect2+pVect1)/2);
            KL = .5 * (sum(pVect1.*(log2(pVect1)-logQvect)) + ...
                sum(pVect2.*(log2(pVect2)-logQvect)));
        case 'sym',
            KL1 = sum(pVect1 .* (log2(pVect1)-log2(pVect2)));
            KL2 = sum(pVect2 .* (log2(pVect2)-log2(pVect1)));
            KL = (KL1+KL2)/2;
            
        otherwise
            error(['Last argument' ' "' varargin{1} '" ' 'not recognized.'])
    end
else
    KL = sum(pVect1 .* (log2(pVect1)-log2(pVect2)));
end
end

function [E] = gdf(V1, V2)
E = norm(V1 - V2, 2);
end