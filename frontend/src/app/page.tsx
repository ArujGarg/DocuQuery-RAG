'use client';

import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Upload, Send, FileText, Loader2, Bot, User, RefreshCw } from 'lucide-react';
import { ChatResponse, Message, UploadResponse } from './types/chat';
import ReactMarkdown from 'react-markdown';


const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [uploading, setUploading] = useState<boolean>(false);
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState<string>('');
  const [asking, setAsking] = useState<boolean>(false);

  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat window on new messages
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, asking]);

  // Handle Document Upload (Cumulative Session)
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    // Send existing session_id if available to append documents cumulatively
    if (sessionId) {
      formData.append('session_id', sessionId);
    }

    try {
      const response = await axios.post<UploadResponse>(`${API_BASE_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const { session_id, filename, chunks_processed } = response.data;
      
      setSessionId(session_id);
      setUploadedFiles((prev) => [...prev, filename]);

      setMessages((prev) => [
        ...prev,
        {
          sender: 'bot',
          text: `Document "${filename}" processed (${chunks_processed} chunks). You can now ask questions across your uploaded context!`,
        },
      ]);
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || 'Failed to upload and index document.';
      alert(errorMsg);
    } finally {
      setUploading(false);
      e.target.value = ''; // Reset input selection
    }
  };

  // Handle Multi-Turn Chat
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || !sessionId) return;

    const userMessage = inputQuery.trim();
    setInputQuery('');

    // Append user message immediately
    setMessages((prev) => [...prev, { sender: 'user', text: userMessage }]);
    setAsking(true);

    try {
      const response = await axios.post<ChatResponse>(`${API_BASE_URL}/chat`, {
        session_id: sessionId,
        question: userMessage,
      });

      setMessages((prev) => [
        ...prev,
        { sender: 'bot', text: response.data.answer },
      ]);
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || 'Error retrieving response from assistant.';
      setMessages((prev) => [
        ...prev,
        { sender: 'bot', text: `⚠️ Error: ${errorMsg}` },
      ]);
    } finally {
      setAsking(false);
    }
  };

  // Reset Session
  const handleResetSession = () => {
    setSessionId(null);
    setUploadedFiles([]);
    setMessages([]);
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white font-sans">
      {/* Sidebar - Upload & Active Documents */}
      <aside className="w-1/3 bg-gray-800 p-6 flex flex-col justify-between border-r border-gray-700">
        <div>
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-2xl font-bold text-indigo-400">DocuQuery RAG</h1>
            {sessionId && (
              <button
                onClick={handleResetSession}
                title="New Session"
                className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-gray-300 transition"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            )}
          </div>
          <p className="text-xs text-gray-400 mb-6">
            Upload PDFs, DOCX, XLSX, or CSVs. Upload multiple files into the same session to query across them.
          </p>

          {/* Upload Dropzone */}
          <div className="border-2 border-dashed border-gray-600 hover:border-indigo-500 rounded-xl p-6 text-center transition bg-gray-800/50">
            <input
              type="file"
              id="fileInput"
              className="hidden"
              onChange={handleFileUpload}
              accept=".pdf,.docx,.doc,.txt,.xlsx,.xls,.csv"
              disabled={uploading}
            />
            <label htmlFor="fileInput" className="cursor-pointer flex flex-col items-center">
              <Upload className="w-10 h-10 text-indigo-400 mb-2" />
              <span className="text-sm font-medium">Click to upload document</span>
              <span className="text-xs text-gray-500 mt-1">PDF, DOCX, XLSX, TXT, CSV</span>
            </label>
          </div>

          {uploading && (
            <div className="flex items-center space-x-2 mt-4 text-indigo-400">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span className="text-sm">Embedding & Indexing...</span>
            </div>
          )}

          {/* List of Active Files in Current Session */}
          {uploadedFiles.length > 0 && (
            <div className="mt-6">
              <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                Session Documents ({uploadedFiles.length})
              </h2>
              <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                {uploadedFiles.map((fname, idx) => (
                  <div
                    key={idx}
                    className="flex items-center space-x-2 bg-gray-700/60 p-2.5 rounded-lg border border-gray-600 text-sm"
                  >
                    <FileText className="w-4 h-4 text-indigo-400 flex-shrink-0" />
                    <span className="truncate text-gray-200">{fname}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="text-xs text-gray-500 text-center border-t border-gray-700 pt-4">
          Next.js App Router • FastAPI • LangChain • ChromaDB
        </div>
      </aside>

      {/* Main Chat Interface */}
      <main className="w-2/3 flex flex-col justify-between p-6">
        {/* Chat Messages */}
        <div ref={chatContainerRef} className="flex-1 overflow-y-auto space-y-4 pr-3">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-500 space-y-2">
              <Bot className="w-12 h-12 text-gray-600" />
              <p className="text-sm">Upload a document from the left sidebar to start chatting.</p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div
                key={index}
                className={`flex items-start space-x-3 ${
                  msg.sender === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {msg.sender === 'bot' && (
                  <div className="p-2 bg-indigo-600 rounded-lg flex-shrink-0">
                    <Bot className="w-5 h-5 text-white" />
                  </div>
                )}
                <div
  className={`p-4 rounded-xl max-w-xl text-sm leading-relaxed ${
    msg.sender === 'user'
      ? 'bg-indigo-600 text-white rounded-br-none'
      : 'bg-gray-800 text-gray-200 border border-gray-700 rounded-bl-none'
  }`}
>
  {msg.sender === 'bot' ? (
    <div className="prose prose-invert max-w-none text-sm leading-relaxed">
      <ReactMarkdown>{msg.text}</ReactMarkdown>
    </div>
  ) : (
    msg.text
  )}
</div>
                {msg.sender === 'user' && (
                  <div className="p-2 bg-gray-700 rounded-lg flex-shrink-0">
                    <User className="w-5 h-5 text-white" />
                  </div>
                )}
              </div>
            ))
          )}
          {asking && (
            <div className="flex items-center space-x-2 text-gray-400 pl-2">
              <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
              <span className="text-xs">Generating response using history & document context...</span>
            </div>
          )}
        </div>

        {/* Input Form */}
        <form onSubmit={handleSendMessage} className="mt-4 flex space-x-3">
          <input
            type="text"
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            disabled={!sessionId || asking}
            placeholder={
              sessionId
                ? 'Ask a question or follow-up...'
                : 'Upload a document first to start asking questions'
            }
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-indigo-500 disabled:opacity-50 text-white placeholder-gray-500 transition"
          />
          <button
            type="submit"
            disabled={!sessionId || asking || !inputQuery.trim()}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white px-5 py-3 rounded-xl transition flex items-center justify-center"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </main>
    </div>
  );
}