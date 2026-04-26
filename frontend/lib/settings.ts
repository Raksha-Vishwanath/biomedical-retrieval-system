import typed from "@/lib/settings.json";

type RawSettings = {
  project: {
    name: string;
    tagline: string;
    research_question: string;
  };
  retrieval_methods: {
    id: string;
    label: string;
    family: string;
    description: string;
  }[];
  analysis_modules: string[];
  frontend: {
    default_query: string;
    results_per_method: number;
  };
};

export async function loadSharedSettings() {
  const settings = typed as RawSettings;

  return {
    project: {
      name: settings.project.name,
      tagline: settings.project.tagline,
      researchQuestion: settings.project.research_question
    },
    retrievalMethods: settings.retrieval_methods,
    analysisModules: settings.analysis_modules,
    frontend: settings.frontend
  };
}
