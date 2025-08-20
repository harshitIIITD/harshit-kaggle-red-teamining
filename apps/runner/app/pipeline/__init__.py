# ABOUTME: Functional pipeline architecture for red-teaming system
# ABOUTME: Replaces agent-based system with composable functional pipelines

from typing import Any, Callable, Dict, List, Optional, TypeVar, Awaitable
from dataclasses import dataclass
import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Type definitions for pipeline components
T = TypeVar('T')
U = TypeVar('U')

PipelineFunction = Callable[[T], Awaitable[U]]
SyncPipelineFunction = Callable[[T], U]

@dataclass
class PipelineContext:
    """Shared context that flows through pipeline stages"""
    run_id: str
    metadata: Dict[str, Any]
    intermediate_results: Dict[str, Any]
    
    def set_result(self, stage_name: str, result: Any) -> None:
        """Store intermediate result from a pipeline stage"""
        self.intermediate_results[stage_name] = result
    
    def get_result(self, stage_name: str, default: Any = None) -> Any:
        """Retrieve result from a previous pipeline stage"""
        return self.intermediate_results.get(stage_name, default)


class PipelineStage(ABC):
    """Abstract base class for all pipeline stages"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def process(self, data: Any, context: PipelineContext) -> Any:
        """Process data through this pipeline stage"""
        pass
    
    async def __call__(self, data: Any, context: PipelineContext) -> Any:
        """Make stage callable with logging"""
        logger.debug(f"Processing stage: {self.name}")
        try:
            result = await self.process(data, context)
            context.set_result(self.name, result)
            return result
        except Exception as e:
            logger.error(f"Error in stage {self.name}: {e}")
            raise


class FunctionalPipeline:
    """
    Revolutionary functional pipeline that replaces the agent-based architecture.
    Data flows through composable stages with immutable transformations.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.stages: List[PipelineStage] = []
        self.parallel_stages: Dict[str, List[PipelineStage]] = {}
    
    def add_stage(self, stage: PipelineStage) -> 'FunctionalPipeline':
        """Add a sequential stage to the pipeline"""
        self.stages.append(stage)
        return self
    
    def add_parallel_stages(self, group_name: str, stages: List[PipelineStage]) -> 'FunctionalPipeline':
        """Add parallel stages that execute concurrently"""
        self.parallel_stages[group_name] = stages
        return self
    
    async def execute(self, initial_data: Any, context: PipelineContext) -> Any:
        """Execute the complete pipeline"""
        logger.info(f"Executing pipeline: {self.name}")
        
        current_data = initial_data
        
        # Process sequential stages
        for stage in self.stages:
            current_data = await stage(current_data, context)
        
        # Process parallel stage groups
        for group_name, parallel_stages in self.parallel_stages.items():
            logger.debug(f"Processing parallel group: {group_name}")
            tasks = [stage(current_data, context) for stage in parallel_stages]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine parallel results
            parallel_results = {}
            for stage, result in zip(parallel_stages, results):
                if isinstance(result, Exception):
                    logger.error(f"Parallel stage {stage.name} failed: {result}")
                    raise result
                parallel_results[stage.name] = result
            
            context.set_result(f"parallel_{group_name}", parallel_results)
            # Use the last non-None result as the continuing data
            current_data = next((r for r in results if r is not None), current_data)
        
        return current_data


class ComposablePipeline:
    """
    Utility for composing multiple pipelines into larger workflows.
    Enables building complex attack generation flows from simple components.
    """
    
    def __init__(self):
        self.pipelines: Dict[str, FunctionalPipeline] = {}
        self.dependencies: Dict[str, List[str]] = {}
    
    def register_pipeline(self, name: str, pipeline: FunctionalPipeline, 
                         depends_on: Optional[List[str]] = None) -> None:
        """Register a named pipeline with optional dependencies"""
        self.pipelines[name] = pipeline
        self.dependencies[name] = depends_on or []
    
    def _resolve_execution_order(self) -> List[str]:
        """Resolve execution order based on dependencies using topological sort"""
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node: str):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node}")
            if node in visited:
                return
            
            temp_visited.add(node)
            for dependency in self.dependencies.get(node, []):
                visit(dependency)
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        for pipeline_name in self.pipelines:
            if pipeline_name not in visited:
                visit(pipeline_name)
        
        return result
    
    async def execute_all(self, initial_data: Any, context: PipelineContext) -> Dict[str, Any]:
        """Execute all registered pipelines in dependency order"""
        execution_order = self._resolve_execution_order()
        results = {}
        
        current_data = initial_data
        
        for pipeline_name in execution_order:
            pipeline = self.pipelines[pipeline_name]
            logger.info(f"Executing composed pipeline: {pipeline_name}")
            
            result = await pipeline.execute(current_data, context)
            results[pipeline_name] = result
            
            # Pass result to next pipeline
            current_data = result
        
        return results


# Specialized pipeline stages for red-teaming

class DataTransformStage(PipelineStage):
    """Generic data transformation stage using provided function"""
    
    def __init__(self, name: str, transform_func: Callable[[Any, PipelineContext], Awaitable[Any]]):
        super().__init__(name)
        self.transform_func = transform_func
    
    async def process(self, data: Any, context: PipelineContext) -> Any:
        return await self.transform_func(data, context)


class ConditionalStage(PipelineStage):
    """Conditional execution stage that branches based on predicate"""
    
    def __init__(self, name: str, 
                 predicate: Callable[[Any, PipelineContext], bool],
                 true_stage: PipelineStage,
                 false_stage: Optional[PipelineStage] = None):
        super().__init__(name)
        self.predicate = predicate
        self.true_stage = true_stage
        self.false_stage = false_stage
    
    async def process(self, data: Any, context: PipelineContext) -> Any:
        if self.predicate(data, context):
            return await self.true_stage(data, context)
        elif self.false_stage:
            return await self.false_stage(data, context)
        else:
            return data


class IterativeStage(PipelineStage):
    """Stage that applies another stage iteratively until condition is met"""
    
    def __init__(self, name: str, 
                 inner_stage: PipelineStage,
                 condition: Callable[[Any, PipelineContext], bool],
                 max_iterations: int = 10):
        super().__init__(name)
        self.inner_stage = inner_stage
        self.condition = condition
        self.max_iterations = max_iterations
    
    async def process(self, data: Any, context: PipelineContext) -> Any:
        current_data = data
        iteration = 0
        
        while not self.condition(current_data, context) and iteration < self.max_iterations:
            current_data = await self.inner_stage(current_data, context)
            iteration += 1
        
        return current_data