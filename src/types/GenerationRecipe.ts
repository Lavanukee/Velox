export type GenerationTaskType = 'text-generation' | 'image-generation' | 'image-captioning' | 'classification' | 'conversation-turn';

export interface GenerationConfig {
    role?: 'system' | 'user' | 'assistant';
    model?: string;
    systemPrompt?: string;
    userTemplate?: (string | RecipeBlock)[];
    assistantTemplate?: (string | RecipeBlock)[];
    // ... other task specific configs
}

export type RecipeBlock =
    | { type: 'text'; content: string }
    | { type: 'generator'; prompt: (string | RecipeBlock)[]; model?: string } // Recursive? Or flat for now. User said "compounded". Let's support simple recursion or just text prompts for now to keep UI sane, but structure allows it.
    | { type: 'source_data'; sourceId: string; transform?: string };

export interface GenerationStep {
    id: string;
    type: GenerationTaskType;
    name: string;
    config: GenerationConfig;
}

export type IterationMode = 'sequential' | 'random' | 'shuffled';

export interface InputSource {
    id: string;
    name: string;
    type: 'file' | 'folder';
    path: string;
    filter?: string[];
    iterationMode: IterationMode;
    allowRepetition: boolean; // For random mode
}

export interface GenerationRecipe {
    id: string;
    name: string;
    description: string;
    steps: GenerationStep[];
    outputFormat: 'jsonl' | 'csv' | 'parquet';
    targetCount: number;
    sources: InputSource[];
}

export const DEFAULT_RECIPE: GenerationRecipe = {
    id: 'new-recipe',
    name: 'New Generation Recipe',
    description: 'Generate a synthetic dataset using LLMs',
    steps: [],
    outputFormat: 'jsonl',
    targetCount: 100,
    sources: []
};
