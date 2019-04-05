function [modelo] = trainBayes(dados)

N = size(dados.x, 1);
meansX = [];
meansY = [];

classes = unique(dados.y);
for i = 1 : length(unique(dados.y)),

    % Seleciona apenas as amostras com a classe pretendida
    indx = find(dados.y == classes(i));
    if not(isempty(indx))
        meansX = [meansX; mean(dados.x(indx, :))];
        meansY = [meansY; classes(i)];
    end
    
    
    modelo.aprioriClass(i) = length(indx) / N;
    
    covs{i} = cov(dados.x(indx, :));
end


modelo.meansX = meansX;
modelo.covs = covs;

modelo.covAll = cov(dados.x);

end


// https://github.com/leandrobmarinho/Mestrado/blob/master/code/Classification/bayes/trainBayes.m