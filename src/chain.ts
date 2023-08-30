import { FaissStore } from 'langchain/vectorstores/faiss';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { ConversationSummaryMemory } from "langchain/memory";
import { FirestoreChatMessageHistory } from "langchain/stores/message/firestore";
import { ChatOpenAI } from "langchain/chat_models";
import * as process from "process";
import { createConversationalRetrievalAgent, createRetrieverTool } from 'langchain/agents/toolkits';
import { AgentExecutor, initializeAgentExecutorWithOptions } from 'langchain/agents';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `
You are the AI of SK매직 and your name is 매직.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
Use emoticons if you need them.

{context}

Question: {question}
Helpful answer in Korean:`;

const AGENT_PROMPT = `
You are the AI of SK매직 and your name is 매직.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
Use emoticons if you need them.

Question: {question}
Helpful answer in Korean:`;

export const makeChain = async (vectorStore: FaissStore, sessionId: string, userId: string) => {
    // 모델 생성
    const model = new ChatOpenAI({
        temperature: 0.6,
        modelName: 'gpt-3.5-turbo-0613',
    });
    // Firebase Store 기록 및 내역 조회
    const firestoreHistory = new FirestoreChatMessageHistory({
        collectionName: "chat_history",
        sessionId: sessionId,
        userId: userId,
        config: {
            projectId:
                "skmagic-chatbot-develop",
        },
    });
    /// 채팅 내역 요약 및 메모리 처리
    const bufferMemory = new ConversationSummaryMemory(
        {
            memoryKey: "chat_history",
            llm: model,
            chatHistory: firestoreHistory,
        });
    /// Chain 생성
    return ConversationalRetrievalQAChain.fromLLM(
        model,
        vectorStore.asRetriever(),
        {
            memory: bufferMemory,
            qaTemplate: QA_PROMPT,
            questionGeneratorTemplate: CONDENSE_PROMPT,
            returnSourceDocuments: false,
            verbose: true,
        },
    );
};

export const makeAgentChain = async (vectorStore: FaissStore, sessionId: string, userId: string) => {
    // 모델 생성
    const retriever = vectorStore.asRetriever();
    const model = new ChatOpenAI({
        temperature: 0.6,
        modelName: 'gpt-3.5-turbo-0613',
    });

    // Firebase Store 기록 및 내역 조회
    const firestoreHistory = new FirestoreChatMessageHistory({
        collectionName: "chat_history",
        sessionId: sessionId,
        userId: userId,
        config: {
            projectId:
                "skmagic-chatbot-develop",
        },
    });
    // Agent에서 사용될 Tool
    const tool = createRetrieverTool(retriever, {
        name: "search_skmagic_product",
        description: "Searches and returns documents regarding the SKmagic 제품 정보",
    });

    /// 채팅 내역 요약 및 메모리 처리
    const memory = new ConversationSummaryMemory(
        {
            memoryKey: "chat_history",
            llm: model,
            chatHistory: firestoreHistory,
        });


    const executor = await initializeAgentExecutorWithOptions([tool], model, {
        agentType: "openai-functions",
        memory,
        returnIntermediateSteps: true,
        agentArgs: {
            prefix:
                AGENT_PROMPT,
        },
    });

    /// Chain 생성
    return executor;
};