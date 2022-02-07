# Feazy

A fast way to break down long term goals into bite-sized tasks and keep yourself accountable to doing them.

Right now, it takes tasks in a [database in Notion](https://www.notion.so/kobifelton/89357b5cf7c749d6872a32636375b064?v=8d62d65e02754fbe84e94b2789d55e68) and schedules them. It takes into account deadlines, wait times (e.g., for feedback) and task dependencies.


## Installation

Install [poetry](https://python-poetry.org/) and then: 

```bash
poetry install
source`poetry env info --path`/bin/activate
```

If you are on anything other than a Apple M1 Mac, then ```poetry add ortools```. For Apple M1 Mac, follow the instructions to install [OR-tools from source](https://github.com/google/or-tools/issues/2722#issuecomment-1028221798).

## Running

You need to have a `.env` file with the your notion internal integration key set as `NOTION_TOKEN`. You can set up an internal integration [here](https://www.notion.so/my-integrations). Make sure to share the database with the integration.

To run the scheduling algorithm:

```bash
python run.py
```