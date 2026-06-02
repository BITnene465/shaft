export type AdvancedFilterControl =
  | {
      type: "search";
      id: string;
      label: string;
      value: string;
      placeholder?: string;
      onChange: (value: string) => void;
    }
  | {
      type: "text";
      id: string;
      label: string;
      value: string;
      placeholder?: string;
      onChange: (value: string) => void;
    }
  | {
      type: "number";
      id: string;
      label: string;
      value: string;
      min?: number;
      max?: number;
      step?: number;
      placeholder?: string;
      onChange: (value: string) => void;
    }
  | {
      type: "select";
      id: string;
      label: string;
      value: string;
      values: string[];
      labels?: Record<string, string>;
      onChange: (value: string) => void;
    };

