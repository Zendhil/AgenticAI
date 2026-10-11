# Standard library imports
import logging

# Third party imports
from langchain_ollama import ChatOllama
from pydantic import ValidationError

# App imports
from agentic_ai.common import Salary, JobPosting
from agentic_ai.common.prompt import JOB_POSTING_PROMPT

logger = logging.getLogger(__name__)

# Ollama's `format` param constrains generation to this JSON schema
def extract(text: str) -> JobPosting:
    prompt=JOB_POSTING_PROMPT.format(text)
    llm = ChatOllama(
        model="llama3.1:latest",
        temperature=0.2,
        top_k=10,
        format=JobPosting.model_json_schema(),
        client_kwargs={"timeout": 60}
    )
    try:
        response = llm.invoke(JOB_POSTING_PROMPT)
        print(response)
    except Exception as e:
        logger.error(f"In extract, Raised Exception:{e}")
        raise e
    return JobPosting.model_validate_json(response.content)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    job_description="""
    About the Role



We are seeking a Data Analyst for a remote contract position focused on data management, reporting, database development, and analytics. The role involves maintaining data collection and reporting systems, improving data quality, and developing analytical solutions that help management understand business performance and make informed decisions.


The successful candidate will work with SQL, Python, Microsoft Excel macros, and data analytics tools to develop reports, dashboards, and data visualizations. The position also requires collaboration with stakeholders, coordination of data remediation activities, and the ability to identify potential program risks and recommend appropriate mitigation measures.


Key Responsibilities



• Develop, implement, and maintain data collection, reporting, and engineering drawing systems to improve work process efficiency and keep data systems current.


• Establish data models and program database mapping systems to support data verification and enhancement.


• Use data analytics software, Microsoft Excel macros, and similar tools to generate analytical outputs, statistical reports, and dashboards for management.


• Develop and enhance databases using SQL programming and Python.


• Coordinate data remediation activities across teams and gather requirements from stakeholders.


• Analyze business problems by visualizing data and presenting clear findings through patterns, trends, anomalies, and other relevant insights.


• Contribute to data analysis, data collection, and data visualization activities.


• Participate in user acceptance testing (UAT) to support the evaluation of data solutions.


• Identify program risks and propose appropriate mitigation measures.


Required Qualifications



The original job description does not specify formal education, minimum years of experience, or mandatory certifications. Candidates should review the responsibilities and technical requirements to assess their suitability for the position.


Relevant technical areas identified in the listing include:


• SQL programming for database development and enhancement.


• Python for data-related development and analysis.


• Microsoft Excel macros or similar data analytics tools.


• Data modeling, database mapping, and data verification.


• Data analysis, reporting, dashboard development, and visualization.


• Stakeholder communication and requirements gathering.


Preferred Qualifications / Requirements



The original listing does not identify a separate set of preferred qualifications. Experience relevant to database enhancement, data remediation, statistical reporting, user acceptance testing, and program risk identification aligns with the duties described.


Salary & Benefits



Compensation and benefits information was not provided in the original job description.


Work Arrangement



• Work location: Remote.


• Employment type: Contract.


    """
    extract('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
