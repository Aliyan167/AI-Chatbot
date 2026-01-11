import os
import pandas as pd
from pathlib import Path
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class HRBPAgent:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")

        self.df = self._load_data()
        self.agent = self._create_agent()

    def _load_data(self):
        base_dir = Path(__file__).resolve().parent.parent
        excel_file = base_dir / "Banking Demo File.xlsx"
        csv_file = base_dir / "Banking Demo File.xlsx - Sheet1.csv"

        if excel_file.exists():
            df = pd.read_excel(excel_file, sheet_name=0)
        elif csv_file.exists():
            df = pd.read_csv(csv_file)
        else:
            files = list(base_dir.glob("*.xlsx")) + list(base_dir.glob("*.csv"))
            if not files:
                raise FileNotFoundError("No Excel or CSV file found")
            file = files[0]
            df = pd.read_excel(file) if file.suffix == ".xlsx" else pd.read_csv(file)

        df.columns = df.columns.str.strip()
        return df

    def _create_agent(self):
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=OPENAI_API_KEY
        )

        return create_pandas_dataframe_agent(
            llm=llm,
            df=self.df,
            verbose=False,
            allow_dangerous_code=True,
            agent_type="tool-calling",
            prefix="""
You are an HR Business Partner AI.

CRITICAL TABLE RULES:
- NEVER output ASCII or pipe-text tables
- NEVER use df.to_string()
- If a table is required, output ONLY a clean Markdown table
- Markdown tables must include headers and a separator row
- If the user does NOT ask for a table, DO NOT use a table

RESPONSE STYLE:
- Be concise and professional
- Allow brief explanation ONLY when useful (max 1–2 lines)
- Use numbered lists for rankings (Top 5, etc.)
- Never guess numbers; always compute from the dataframe
- Avoid filler text, greetings, or conclusions
"""
        )

    def ask(self, question: str) -> str:
        q = question.lower()

        # ✅ SAFE PYTHON HANDLING FOR EMPLOYEE COUNT
        if any(k in q for k in [
            "how many employees",
            "total employees",
            "number of employees",
            "employee count"
        ]):
            return f"Total Employees: {len(self.df)}"

        # 🔍 Detect explicit table request
        wants_table = any(k in q for k in [
            "table",
            "tabular",
            "dataframe",
            "show in table",
            "display table"
        ])

        hr_prompt = f"""
You are an HRBP AI Assistant.

INSTRUCTIONS:
- Keep answers short and clear
- Add a brief explanation only if it helps understanding
- {"Use a clean Markdown table." if wants_table else "DO NOT use tables."}
- NEVER output ASCII tables
- Use numbered lists for top or ranked items
- Use only the dataframe for answers

User Question:
{question}

Answer now.
"""
        response = self.agent.invoke({"input": hr_prompt})
        return response.get("output", "")
