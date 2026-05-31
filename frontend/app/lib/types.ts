export type Status = "draft" | "reviewed";

export interface Project {
  slug: string;
  name: string;
  model_name: string;
  record_count: number;
  created_at: string;
  updated_at: string;
}

export interface RecordSummary {
  id: number;
  title: string | null;
  status: Status;
  n_src_chunks: number;
  n_tgt_chunks: number;
  n_pairs: number;
  created_at: string;
  updated_at: string;
}

export interface ChunkedSegment {
  type: "aligned" | "src_only_unaligned" | "tgt_only_unaligned" | "empty";
  src_range: [number, number];
  tgt_range: [number, number];
  src: string[];
  tgt: string[];
}

export interface RecordOut {
  id: number;
  title: string | null;
  src_text: string;
  tgt_text: string;
  src_chunks: string[];
  tgt_chunks: string[];
  gt_pairs: [number, number][];
  model_pairs: [number, number][];
  model_response: string | null;
  status: Status;
  notes: string | null;
  chunked_sets: ChunkedSegment[];
  created_at: string;
  updated_at: string;
}

export interface RecordPatch {
  title?: string | null;
  src_chunks?: string[];
  tgt_chunks?: string[];
  gt_pairs?: [number, number][];
  status?: Status;
  notes?: string | null;
}

export interface InferOut {
  response: string;
  pairs: [number, number][];
  chunked_sets: ChunkedSegment[];
  parse_error: boolean;
}

export interface TranslateSourceOut {
  translations: string[];
  response: string;
  parse_error: boolean;
}

export interface BatchUploadError {
  src_file: string | null;
  tgt_file: string | null;
  detail: string;
}

export interface BatchUploadOut {
  records: RecordOut[];
  errors: BatchUploadError[];
}
