import { TextLoader } from 'langchain/document_loaders/fs/text';
import { Document } from 'langchain/document';
import fs from 'fs/promises';

export async function loadTextDocument(filePath: string): Promise<Document<Record<string, any>>[]> {
    const loader = new TextLoader(filePath);
    const docs = await loader.load();
    return docs;
}

export async function writeObjectToFile(embeddings, fileName): Promise<void> {
    try {
        const jsonString = JSON.stringify(embeddings, null, 2);
        await fs.writeFile(`./data/${fileName}`, jsonString);
        console.log('Successfully wrote JSON to ' + fileName);
    } catch (err) {
        console.error('Error writing to file', err);
    }
}
