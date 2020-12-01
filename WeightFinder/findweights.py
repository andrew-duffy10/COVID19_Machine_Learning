import sys

from WeightFinder.homebrew_agent import HomebrewAgent
from WeightFinder.sklearn_agent import SklearnAgent

USAGE = "Usage: ./findweights -agent <sklearn | homebrew> -state [U.S. state] -d1 <Initial date> -d2 [Ending date]"


def parse_args(argv):
    agent_type = None
    state = None
    start_day = None
    end_day = None
    try:
        for idx, arg in enumerate(argv):
            if arg == "-agent":
                agent_type = argv[idx + 1]
            elif arg == "-state":
                state = argv[idx + 1]
            elif arg == "-d1":
                start_day = argv[idx + 1]
            elif arg == "-d2":
                end_day = argv[idx + 1]
    except Exception:
        print(USAGE)
        sys.exit(1)
    if (agent_type and start_day and not state and not end_day) or (agent_type and start_day and state and end_day):
        return agent_type, state, start_day, end_day
    else:
        print(USAGE)
        sys.exit(1)


def run_agent(agent, state, start_day, end_day):
    if state and start_day and end_day:
        coefficients = agent.run_for_us_state(state, start_day, end_day)
    elif start_day and not state and not end_day:
        coefficients = agent.run_for_day('04-12-2020')
    else:
        print(USAGE)
        sys.exit(1)
    return coefficients


def main(argv):
    agent_type, state, start_day, end_day = parse_args(argv)
    independent_variables = ['Confirmed', 'People_Tested']
    dependent_variable = 'Incident_Rate'
    if agent_type == "sklearn":
        agent = SklearnAgent(independent_variables, dependent_variable)
        coefficients = run_agent(agent, state, start_day, end_day)
    else:
        agent = HomebrewAgent(independent_variables, dependent_variable)
        coefficients = run_agent(agent, state, start_day, end_day)
    print(coefficients)
    return coefficients


if __name__ == "__main__":
    main(sys.argv[1:])
