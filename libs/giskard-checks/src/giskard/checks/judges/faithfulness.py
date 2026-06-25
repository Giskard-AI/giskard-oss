@override
    async def get_inputs(self, trace: TraceType) -> dict[str, str]:
        """Build template variables from resolved inputs.

        Parameters
        ----------
        trace : Trace
            Trace for resolving inputs.

        Returns
        -------
        dict[str, str]
            Template variables with 'answer' and 'source' keys.
        """
        resolved_answer = provided_or_resolve(
            trace,
            key=self.answer_key,
            value=provide_not_none(self.answer),
        )
        resolved_source = provided_or_resolve(
            trace,
            key=self.source_key,
            value=provide_not_none(self.source),
        )
        answer_str = (
            "\n".join(map(str, resolved_answer))
            if isinstance(resolved_answer, list)
            else str(resolved_answer)
        )
        source_str = (
            "\n".join(map(str, resolved_source))
            if isinstance(resolved_source, list)
            else str(resolved_source)
        )
        return {
            "answer": answer_str,
            "source": source_str,
        }