import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { Document } from 'langchain/document';

class EmbeddingModel {
    public model: OpenAIEmbeddings;

    constructor() {
        this.model = new OpenAIEmbeddings({ modelName: 'text-embedding-ada-002' });
    }

    public async GetEmbeddings(documents: Document<Record<string, any>>[]): Promise<number[][]> {
        const content = documents.map((doc) => doc.pageContent);
        const result = await this.model.embedDocuments(content);
        return result;
    }
}

export default EmbeddingModel;