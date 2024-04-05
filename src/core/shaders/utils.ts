export const conditionalCode = (
    condition: boolean,
    shaderCodeA: string,
    shaderCodeB: string = "",
) => {
    return condition ? shaderCodeA : shaderCodeB;
};

export const constants = {
    inf: 1e30,
    infu32: 4294967295,
    pi: 3.1415926535,
    epsilon: 0.001,
    fireflyClamp: 50,
};
