export interface Message {
  sender: 'user' | 'bot';
  text: string;
}

export interface UploadResponse {
  status: string;
  session_id: string;
  filename: string;
  chunks_processed: number;
  message: string;
}

export interface ChatResponse {
  session_id: string;
  question: string;
  answer: string;
}