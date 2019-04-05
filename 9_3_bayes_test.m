function [classes, valores, acoes] = testeBayes(modelo, dados, conf)

    % Probabilidade a priori de X
    aprioriClassX = [];
    for j = 1 : length(modelo.aprioriClass)
        aprioriClassX(j,:) = modelo.aprioriClass(j)*mvnpdf(dados.x, modelo.meansX(j,:), modelo.covs{j})';
    end
    aprioriClassX = sum(aprioriClassX);

    for i = 1 : size(modelo.meansX,1)
        a_posteriori(i, :) = (modelo.aprioriClass(i)*mvnpdf(dados.x, modelo.meansX(i,:), ...
            modelo.covs{i})')./aprioriClassX;
    end

    [valores, classes] = max(a_posteriori);

end



//  https://github.com/leandrobmarinho/Mestrado/raw/master/code/Classification/bayes/testeBayes.m