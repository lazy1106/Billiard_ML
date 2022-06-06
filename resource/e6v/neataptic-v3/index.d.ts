declare namespace neataptic {

    export interface NeatOptions {
        mutation?: IMutation[] | IMutation;
        network?: Network;
        equal?: boolean;
        clear?: boolean;
        popsize?: number;
        elitism?: number;
        provenance?: number;
        mutationRate?: number;
        mutationAmount?: number;
        fitnessPopulation?: boolean;
        selection?: ISelection;
        crossover?: ICrossover | ICrossover[];
        maxNodes?: number;
        maxConns?: number;
        maxGates?: number;
    }

    export type COST_FUNCTION = (target: number, output: number) => number;
    export type RATE_FUNCTION = (gamma?: number, ...rest) => (...rest) => number;
    export type ACTIVATION_FUNCTION = (x: number, derivate: number) => number;
    export type IConnectable = Node | Group | Layer;

    interface IMutation {
        name: string;
        keep_gates?: boolean;
        min?: number;
        max?: number;
        mutateOutput?: boolean;
        allowed?: ACTIVATION_FUNCTION[];
    }

    export interface ICrossover {
        name: string;
        config?: number[];
    }
    export interface IGating {
        name: string;
    }
    export interface IConnection {
        name: string;
    }
    export interface ISelection {
        name: string;
        power?: number;
        probability?: number;
    }

    export const methods: {
        activation: {
            LOGISTIC: ACTIVATION_FUNCTION,
            TANH: ACTIVATION_FUNCTION,
            IDENTITY: ACTIVATION_FUNCTION,
            STEP: ACTIVATION_FUNCTION,
            RELU: ACTIVATION_FUNCTION,
            SOFTSIGN: ACTIVATION_FUNCTION,
            SINUSOID: ACTIVATION_FUNCTION,
            GAUSSIAN: ACTIVATION_FUNCTION,
            BENT_IDENTITY: ACTIVATION_FUNCTION,
            BIPOLAR: ACTIVATION_FUNCTION,
            BIPOLAR_SIGMOID: ACTIVATION_FUNCTION,
            HARD_TANH: ACTIVATION_FUNCTION,
            ABSOLUTE: ACTIVATION_FUNCTION,
            INVERSE: ACTIVATION_FUNCTION,
            SELU: ACTIVATION_FUNCTION,
        },
        crossover: {
            SINGLE_POINT: ICrossover,
            TWO_POINT: ICrossover,
            UNIFORM: ICrossover,
            AVERAGE: ICrossover,
        },
        gating: {
            OUTPUT: IGating,
            INPUT: IGating,
            SELF: IGating,
        },
        connection: {
            ALL_TO_ALL: IConnection,
            ALL_TO_ELSE: IConnection,
            ONE_TO_ONE: IConnection,
        },
        mutation: {
            ADD_NODE: IMutation,
            SUB_NODE: IMutation,
            ADD_CONN: IMutation,
            SUB_CONN: IMutation,
            MOD_WEIGHT: IMutation,
            MOD_BIAS: IMutation,
            MOD_ACTIVATION: IMutation,
            ADD_SELF_CONN: IMutation,
            SUB_SELF_CONN: IMutation,
            ADD_GATE: IMutation,
            SUB_GATE: IMutation,
            ADD_BACK_CONN: IMutation,
            SUB_BACK_CONN: IMutation,
            SWAP_NODES: IMutation,
    
            ALL: IMutation[];
            FFW: IMutation[];
        },
        cost: {
            CROSS_ENTROPY: COST_FUNCTION,
            MSE: COST_FUNCTION,
            BINARY: COST_FUNCTION,
            MAE: COST_FUNCTION,
            MAPE: COST_FUNCTION,
            MSLE: COST_FUNCTION,
            HINGE: COST_FUNCTION,
        },
        rate: {
            FIXED: RATE_FUNCTION,
            STEP: RATE_FUNCTION,
            EXP: RATE_FUNCTION,
            INV: RATE_FUNCTION,
        },
        selection: {
            FITNESS_PROPORTIONATE: ISelection,
            POWER: ISelection,
            TOURNAMENT: ISelection,
        },
    }

    export const architect: {
        Construct: (list: IConnectable[]) => Network;
        Perceptron: (...layers) => Network;
        Random: (inputSize: number, hiddenSize: number, outputSize: number, options?: {
            connections?: number;
            backConnections?: number;
            selfconnections?: number;
            gates?: number;
        }) => Network;
        LSTM: (...layers) => Network;
        GRU: (...layers) => Network;
        Hopfield: (size: number) => Network;
        NARX: (inputSize: number, hiddenLayers: number, outputSize: number, previousInput: number, previousOutput: number) => Network;
    }

    export class Group {
        constructor(size: number);
        activate(value: number[]): number[];
        propagate(rate: number, momentum: number, target: IConnectable[]): void;
        connect(target: IConnectable, method: IConnection, weight: number): Connection[];
        gate(connections: Connection[], method: IConnection): void;
        set(values: any): void;
        disconnect(target: IConnectable, twosided: boolean): void;
        clear(): void;
    }

    export class Layer {
        constructor();
        activate(value: number[]): number[];
        propagate(rate: number, momentum: number, target: IConnectable[]): void;
        connect(target: IConnectable, method: IConnection, weight: number): Connection[];
        gate(connections: Connection[], method: IConnection): void;
        set(values: any): void;
        disconnect(target: IConnectable, twosided: boolean): void;
        clear(): void;

        static Dense(size: number): Layer;
        static LSTM(size: number): Layer;
        static GRU(size: number): Layer;
        static Memory(size: number): Layer;
    }

    export class Node {
        constructor(type: string);
        activate(input?: number): number;
        noTraceActivate(input?: number): number;
        propagate(rate: number, momentum: number, update: boolean, target: Node): void;
        connect(target: Node, weight: number): Connection[];
        disconnect(node: Node, twosided: boolean): void;
        gate(connections: Connection[]): void;
        ungate(connections: Connection[]): void;
        clear(): void;
        mutate(method: RATE_FUNCTION): void;
        isProjectingTo(node: Node): boolean;
        isProjectedBy(node: Node): boolean;
        toJSON(): any;
        static fromJSON(json: any): Node;
    }

    export class Connection {
        constructor(from: IConnectable, to: IConnectable, weight: number);
        toJSON(): any;
        static innovationID(a: number, b: number): number;
    }

    export class Network {
        score: number;
        constructor(inputSize: number, outputSize: number);
        activate(input: number[], training?: boolean): number[];
        noTraceActivate(input: number[]): number[];
        propagate(rate: number, momentum: number, update: boolean, target: Node[]): void;
        clear(): void;
        connect(from: Node, to: Node, weight: number): Connection[];
        disconnect(from: Node, to: Node): void;
        gate(node: Node, connection: Connection): void;
        ungate(connection: Connection): void;
        remove(node: Node): void;
        mutate(method: RATE_FUNCTION): void;
        train(set: any, options: any): { error: any, iterations: number, time: number, };
        test(set: any, cost?: COST_FUNCTION): boolean;
        graph(width: number, height: number): any;
        toJSON(): any;
        set(values: any): void;
        evolve(set: any, options: any): Promise<{ error: number, iterations: number, time: number, }>;
        standalone(): string;
        serialize(): any;
        static fromJSON(json: any): Network;
        static merge(network1: Network, network2: Network): Network;
        static crossOver(network1: Network, network2: Network, equal: boolean): Network;
    }

    export class Neat {
        population: Network[];
        generation: number;

        constructor(inputSize: number, outputSize: number, fitnessFunction: Function, options: NeatOptions);
        createPool(network: Network): void;
        evolve(): Promise<Network>;
        getOffspring(): Promise<Network>;
        selectMutationMethod(genome: Network): IMutation;
        mutate(): void;
        evaluate(): Promise<void>;
        sort(): void;
        getFittest(): Network;
        getAverage(): number;
        getParent(): Network;
        export(): any;
        import(json: any): void;
    }
}