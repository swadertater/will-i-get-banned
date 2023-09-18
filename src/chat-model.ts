import { OpenAIChat } from 'langchain/llms/openai';

class ChatModel {
    public model: OpenAIChat;

    constructor(modelName: string) {
        this.model = new OpenAIChat({ modelName });
    }
}

export default ChatModel;